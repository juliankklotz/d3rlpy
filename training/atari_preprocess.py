#!/usr/bin/env python3
"""Standard Atari preprocessing for the minari Pong dataset.

Converts raw minari frames [3, 210, 160] uint8 RGB into the standard offline-RL
Atari format [84, 84] uint8 grayscale, then relies on d3rlpy's FrameStack
picker/slicer to stack 4 frames at sampling time -> [4, 84, 84].

Why: raw [3,210,160] is ~15x the pixels of [4,84,84] and measured >0.24 s/step
on an A40 (~33h for 500k steps), which does not fit a 12h SLURM job. The
[4,84,84] format is what the Decision Transformer paper and the offline-Atari
literature use, and what this thesis describes.

Uses torch for the resize (already a dependency; no cv2/d4rl-atari needed —
d4rl-atari's atari-py dependency fails to build without cmake on the cluster).

Preprocessing follows the standard Atari pipeline:
  1. RGB -> grayscale (ITU-R 601-2 luma: 0.299R + 0.587G + 0.114B)
  2. bilinear resize 210x160 -> 84x84
  3. keep uint8 (PixelObservationScaler divides by 255 at train time)
Frame stacking is NOT done here — it is applied by FrameStackTransitionPicker /
FrameStackTrajectorySlicer at sample time, which avoids materialising 4x the data.
"""
from typing import Optional

import numpy as np


def rgb_to_gray84(obs: np.ndarray, size: int = 84) -> np.ndarray:
    """[N, 3, H, W] uint8 RGB -> [N, 1, size, size] uint8 grayscale (channel-first).

    The singleton channel axis is REQUIRED: d3rlpy's FrameStack picker stacks
    frames along axis 0 of each observation, so per-frame obs must be
    [1,size,size] to yield [num_stack,size,size] after stacking. Returning a
    bare [size,size] instead produces [num_stack*size, size] (wrong).

    Args:
        obs: raw frames, channel-first uint8 RGB.
        size: output square resolution (84 is the Atari standard).

    Returns:
        uint8 array [N, 1, size, size].
    """
    import torch

    assert obs.ndim == 4 and obs.shape[1] == 3, f"expected [N,3,H,W], got {obs.shape}"

    x = torch.from_numpy(np.ascontiguousarray(obs)).float()  # [N,3,H,W]
    # ITU-R 601-2 luma transform (same as PIL's "L" conversion)
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])  # [N,H,W]
    gray = gray.unsqueeze(1)  # [N,1,H,W]
    resized = torch.nn.functional.interpolate(
        gray, size=(size, size), mode="bilinear", align_corners=False
    )  # [N,1,size,size]
    # keep the channel axis (see docstring): FrameStack stacks on axis 0
    out = resized.clamp(0, 255).to(torch.uint8).numpy()
    return out


def preprocess_atari_buffer(
    buffer,
    size: int = 84,
    num_stack: int = 4,
    verbose: bool = True,
):
    """Return a new ReplayBuffer with [size,size] grayscale obs + frame stacking.

    Args:
        buffer: ReplayBuffer whose episodes hold [T,3,210,160] uint8 observations.
        size: output resolution (84 standard).
        num_stack: frames stacked at sampling time via d3rlpy's FrameStack
            picker/slicer -> effective obs [num_stack, size, size].
        verbose: print progress/shape info.

    Returns:
        New ReplayBuffer with downsampled observations and frame-stacking
        picker/slicer attached.
    """
    from d3rlpy.dataset import (
        Episode,
        FIFOBuffer,
        ReplayBuffer,
        FrameStackTransitionPicker,
        FrameStackTrajectorySlicer,
    )

    eps = buffer.episodes
    if verbose:
        print(f"Preprocessing {len(eps)} episodes: RGB[3,210,160] -> gray[1,{size},{size}] ...")

    new_eps = []
    for i, ep in enumerate(eps):
        obs = ep.observations
        assert isinstance(obs, np.ndarray), "expected ndarray observations"
        gray = rgb_to_gray84(obs, size=size)
        new_eps.append(
            Episode(
                observations=gray,
                actions=ep.actions,
                rewards=ep.rewards,
                terminated=ep.terminated,
            )
        )
        if verbose and (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(eps)} episodes")

    info = buffer.dataset_info
    new_buffer = ReplayBuffer(
        buffer=FIFOBuffer(limit=sum(len(e) for e in new_eps)),
        episodes=new_eps,
        action_space=info.action_space,
        action_size=info.action_size,
        transition_picker=FrameStackTransitionPicker(n_frames=num_stack),
        trajectory_slicer=FrameStackTrajectorySlicer(n_frames=num_stack),
    )
    if verbose:
        s = new_eps[0].observations.shape
        print(f"✓ preprocessed: per-frame {s}, stacked at sample time -> "
              f"[{num_stack},{size},{size}]")
    return new_buffer


class GrayResizeObservation:
    """Gym wrapper: RGB frame -> [1,size,size] uint8 grayscale, via rgb_to_gray84.

    Deliberately reuses the SAME function the dataset preprocessing uses, so the
    policy sees identically-processed observations at train and eval time. Using
    d3rlpy's cv2-based AtariPreprocessing instead would risk a subtle train/eval
    mismatch (different interpolation kernel and grayscale weights), on top of
    requiring opencv.

    NOTE: unlike d3rlpy's AtariPreprocessing, this does NOT do frame-skip,
    noop-starts, or max-pooling over consecutive frames. It only matches the
    observation transform. Stack with d3rlpy's FrameStack for [num_stack,size,size].
    """

    def __init__(self, env, size: int = 84):
        import gymnasium
        from gymnasium.spaces import Box

        self.env = env
        self._size = size
        # NOTE: emit [size,size] WITHOUT a channel axis here. d3rlpy's FrameStack
        # stacks along a NEW leading axis, so [size,size] -> [num_stack,size,size],
        # matching the dataset. Emitting [1,size,size] would give
        # [num_stack,1,size,size] (wrong rank).
        self.observation_space = Box(
            low=0, high=255, shape=(size, size), dtype=np.uint8
        )
        self.action_space = env.action_space
        # expose the rest of the gym API by delegation
        self.metadata = getattr(env, "metadata", {})
        self.spec = getattr(env, "spec", None)

    def _obs(self, obs):
        arr = np.asarray(obs)
        # accept [H,W,3] (gym default) or [3,H,W] (channel-first)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = np.transpose(arr, (2, 0, 1))
        assert arr.ndim == 3 and arr.shape[0] == 3, f"expected RGB frame, got {arr.shape}"
        # rgb_to_gray84 returns [N,1,size,size]; drop batch AND channel axes so
        # FrameStack yields [num_stack,size,size] (see __init__ note).
        return rgb_to_gray84(arr[None, ...], size=self._size)[0, 0]  # [size,size]

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):  # gymnasium: (obs, info)
            obs, info = out
            return self._obs(obs), info
        return self._obs(out)

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:  # gymnasium: obs, reward, terminated, truncated, info
            obs, r, term, trunc, info = out
            return self._obs(obs), r, term, trunc, info
        obs, r, done, info = out  # legacy gym
        return self._obs(obs), r, done, info

    def render(self, *a, **kw):
        return self.env.render(*a, **kw)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        # delegate anything else (e.g. unwrapped, np_random) to the wrapped env
        return getattr(self.env, name)


def make_atari_eval_env(raw_env, size: int = 84, num_stack: int = 4):
    """Wrap a raw Atari env so its observations match the preprocessed dataset.

    Applies the same grayscale+resize as the dataset, then d3rlpy's FrameStack,
    yielding [num_stack, size, size] uint8 — identical to what the policy is
    trained on.
    """
    from d3rlpy.envs import FrameStack

    return FrameStack(GrayResizeObservation(raw_env, size=size), num_stack=num_stack)

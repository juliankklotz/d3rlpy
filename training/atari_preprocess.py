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

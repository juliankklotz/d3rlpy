import dataclasses
from typing import Optional, Union, Tuple, Literal


import torch
from torch import nn
from ...logging import LOG
from ...algos.transformer.base import TransformerAlgoImplBase
from ...algos.qlearning.base import QLearningAlgoImplBase
from ...algos.qlearning.torch.utility import (
    ContinuousQFunctionMixin,
    DiscreteQFunctionMixin,
)
from ...models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
)
from ...optimizers import OptimizerWrapper
from ...torch_utility import Modules, TorchMiniBatch, TorchTrajectoryMiniBatch, hard_sync
from ...types import Shape

__all__ = ["FQEBaseImpl", "FQEImpl", "DiscreteFQEImpl", "FQEBaseModules","FQETrajectoryBaseImpl","FQETrajectoryImpl","DiscreteFQETrajectoryImpl","dt_predict_next_actions","dt_predict_next_action_probs"]


def dt_predict_next_actions(
    algo: TransformerAlgoImplBase,                                   #
    traj: TorchTrajectoryMiniBatch,
    target_rtg: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Strictly-causal next-action prediction for FQE with a Decision Transformer.

    Key properties:
    - Single forward pass with teacher forcing (dataset actions), relying on the DT's
      internal causal attention (lower-triangular) for no future leakage.
    - Preds are (B, L, A), zeroed on padded steps.
    - next_actions are **dense** (B*(L-1), A) to match TorchTrajectoryMiniBatch.to_transition_batch()
      when using fixed-length trajectory sampling; rows that cross padding are zeroed (but kept).
    - Both outputs are contiguous.

    Args:
        algo_or_impl: d3rlpy DecisionTransformer algo or its ._impl
        traj: TorchTrajectoryMiniBatch with fields:
              observations: (B, L, *obs)
              actions:      (B, L, A)
              returns_to_go:(B, L, 1) or (B, L)
              timesteps:    (B, L, 1) or (B, L)
              masks:        (B, L) or (B, L, 1) with 1=valid, 0=pad
        target_rtg: If given, overrides the RTG channel with this constant
            value at every timestep instead of using the trajectory's own
            logged return-to-go. This evaluates the policy as it would behave
            if told to aim for a fixed deployment-time target, rather than
            replaying the behaviour policy's actual future return.

    Returns:
        preds:        (B, L, A) predictions, zeroed on padded steps, contiguous
        next_actions: (B*(L-1), A) flattened, zeroed where (t,t+1) crosses padding, contiguous
    """
    # Resolve impl and DT module
    impl = algo
    transformer = impl._modules.transformer

    obs   = traj.observations
    acts  = traj.actions
    rtg   = traj.returns_to_go
    steps = traj.timesteps
    masks = traj.masks  # 1=valid, 0=pad

    if target_rtg is not None:
        rtg = torch.full_like(rtg, float(target_rtg))

    # Build pad mask as used in training (1=pad, 0=keep)
    pad_mask = 1 - masks

    # Forward pass (teacher forcing) — underlying GPT-style model is causal.
    # DiscreteDecisionTransformer returns (probs, logits); use logits for predictions
    with torch.no_grad():
        _, preds = transformer(obs, acts, rtg, steps, pad_mask)  # (B, L, A)

    # Harmonize mask shapes
    if masks.dim() == 3 and masks.size(-1) == 1:
        mask2d = masks.squeeze(-1)           # (B, L)
        mask3d = masks                       # (B, L, 1)
    elif masks.dim() == 2:
        mask2d = masks                        # (B, L)
        mask3d = masks.unsqueeze(-1)          # (B, L, 1)
    else:
        raise ValueError("traj.masks must have shape (B, L) or (B, L, 1).")

    # Zero out padded predictions and make contiguous
    preds = (preds * mask3d).contiguous()     # (B, L, A)

    # Build dense next-action matrix aligned with to_transition_batch():
    # Always output (B, L-1, A) → flatten to (B*(L-1), A).
    # If either t or t+1 is padded, zero that row (but keep it to preserve length).
    B, L = mask2d.shape
    assert preds.shape[0] == B and preds.shape[1] == L

    # Predictions at next step indices
    preds_next = preds[:, 1:, :]              # (B, L-1, A)

    mask2d_bool = mask2d.to(torch.bool)

    # Valid pair mask (both sides valid)
    pair_keep = (mask2d_bool[:, :-1] & mask2d_bool[:, 1:]).unsqueeze(-1)  # (B, L-1, 1), bool

    # Zero out invalid pairs, but keep the rows
    preds_next = preds_next * pair_keep.to(preds_next.dtype)

    # Flatten to match to_transition_batch() shape and ensure contiguity
    next_actions = preds_next.reshape(B * (L - 1), preds_next.size(-1)).contiguous()

    return preds, next_actions

def dt_predict_next_action_probs(
    algo: TransformerAlgoImplBase,
    traj: TorchTrajectoryMiniBatch,
    target_rtg: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Softmax action probabilities for FQE expected-Q bootstrap with discrete DT.

    Returns the policy's probability distribution over next-state actions,
    for use in expected-SARSA-style bootstrap: y = r + γ·Σ_a' π(a'|s')·Q(s',a').

    Args:
        algo: d3rlpy DiscreteDecisionTransformer impl
        traj: TorchTrajectoryMiniBatch
        target_rtg: Optional fixed RTG override

    Returns:
        preds_probs: (B, L, A) softmax probabilities, zeroed on padded steps
        next_action_probs: (B*(L-1), A) flattened probabilities, aligned with to_transition_batch()
    """
    impl = algo
    transformer = impl._modules.transformer

    obs   = traj.observations
    acts  = traj.actions
    rtg   = traj.returns_to_go
    steps = traj.timesteps
    masks = traj.masks

    if target_rtg is not None:
        rtg = torch.full_like(rtg, float(target_rtg))

    pad_mask = 1 - masks

    with torch.no_grad():
        probs, _ = transformer(obs, acts, rtg, steps, pad_mask)  # (B, L, A), already softmax

    # Harmonize mask shapes
    if masks.dim() == 3 and masks.size(-1) == 1:
        mask2d = masks.squeeze(-1)
        mask3d = masks
    elif masks.dim() == 2:
        mask2d = masks
        mask3d = masks.unsqueeze(-1)
    else:
        raise ValueError("traj.masks must have shape (B, L) or (B, L, 1).")

    # Zero out padded probabilities
    preds_probs = (probs * mask3d).contiguous()  # (B, L, A)

    # Build next-action probabilities aligned with to_transition_batch()
    B, L = mask2d.shape
    probs_next = preds_probs[:, 1:, :]  # (B, L-1, A)

    mask2d_bool = mask2d.to(torch.bool)
    pair_keep = (mask2d_bool[:, :-1] & mask2d_bool[:, 1:]).unsqueeze(-1)  # (B, L-1, 1)

    probs_next = probs_next * pair_keep.to(probs_next.dtype)
    next_action_probs = probs_next.reshape(B * (L - 1), probs_next.size(-1)).contiguous()

    return preds_probs, next_action_probs

@torch.no_grad()
def dt_predict_autoreg(
    algo: TransformerAlgoImplBase,                                   # 
    traj: TorchTrajectoryMiniBatch,       # (B,L,...) batch
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Strictly-causal DT inference for FQE on TorchTrajectoryMiniBatch.

    Returns
    -------
    preds        : (B, L, A) per-timestep DT predictions (zero on pads)
    next_actions : (N_valid, A) flattened, aligned with traj.to_transition_batch()
                   (last steps in episodes get zeros -> no bootstrap via terminals)
    """
    impl        = algo
    transformer = impl._modules.transformer
    dev         = impl.device

    obs   = traj.observations.to(dev)   # (B,L,...)
    acts  = traj.actions.to(dev)        # (B,L,A)
    rtg   = traj.returns_to_go.to(dev)  # (B,L,1) already causal
    tstep = traj.timesteps.to(dev)      # (B,L,1) or (B,L)
    masks = traj.masks.to(dev).bool()   # (B,L) True=valid

    B, L = masks.shape
    A    = acts.size(-1)

    preds = torch.zeros((B, L, A), device=dev, dtype=acts.dtype)

    # Causal, dataset teacher-forcing loop (prefix-only)
    for t in range(L):
        cur = t + 1
        attention_mask = (~masks[:, :cur]).to(obs.dtype)  # same convention as training (1=pad)

        # Right-shift actions inline: feed a_{0..t-1} to predict a_t
        shifted = torch.zeros((B, cur, A), device=dev, dtype=acts.dtype)
        # safe even when t==0 (slice is empty)
        shifted[:, 1:cur] = acts[:, :t]

        out = transformer(
            obs[:, :cur],
            shifted,
            rtg[:, :cur],
            tstep[:, :cur],
            attention_mask,
        )  # (B,cur,A)

        preds[:, t] = out[:, -1]  # take prediction for position t

    # Zero-out pads defensively
    preds = preds * masks.unsqueeze(-1)

    # Build t+1 actions, aligned to (b,t) rows; last steps -> zeros
    next_actions_all = torch.zeros_like(preds)
    next_actions_all[:, :-1] = preds[:, 1:]     # shift left by 1 along time

    # Flatten to match to_transition_batch() (valid rows only)
    next_actions = next_actions_all[masks].contiguous()         # (N_valid, A)

    return preds, next_actions

@torch.no_grad()
def dt_predict_next_actions_stateful(
    algo: TransformerAlgoImplBase,                                   # 
    traj: TorchTrajectoryMiniBatch,       # (B,L,...) batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Strictly causal DT inference for FQE on TorchTrajectoryMiniBatch.

    Returns
    -------
    preds        : (B, L, A) per-timestep predictions (zeroed on pads), contiguous
    next_actions : (N_valid, A) flattened, aligned row-major with valid (b,t), contiguous
                   (last steps per episode get zeros → no bootstrap via terminals)
    """
    # accept either algo or impl
    impl = algo.impl if hasattr(algo, "impl") and algo.impl is not None else algo
    transformer = impl._modules.transformer
    dev = impl.device

    # bring inputs to device (also contiguous up-front)
    obs   = traj.observations.to(dev).contiguous()   # (B,L,...)
    acts  = traj.actions.to(dev).contiguous()        # (B,L,A)
    rtg   = traj.returns_to_go.to(dev).contiguous()  # (B,L,1) causal
    tstep = traj.timesteps.to(dev).contiguous()      # (B,L,1) or (B,L)
    masks = traj.masks.to(dev).bool().contiguous()   # (B,L)

    B, L = masks.shape
    A    = acts.size(-1)

    preds = torch.zeros((B, L, A), device=dev, dtype=acts.dtype)

    # --- per-row, prefix-only, dataset teacher forcing ---
    for b in range(B):
        valid_len = int(masks[b].sum().item())
        for t in range(valid_len):
            cur = t + 1

            # attention mask for this row/prefix: 1=pad, 0=keep (matches your training)
            attn = (~masks[b, :cur]).to(obs.dtype).unsqueeze(0)

            # right-shift actions: feed a_{0..t-1} to predict a_t
            shifted = torch.zeros((1, cur, A), device=dev, dtype=acts.dtype)
            if t > 0:
                shifted[:, 1:cur] = acts[b, :t].unsqueeze(0)

            out = transformer(
                obs[b, :cur].unsqueeze(0),
                shifted,
                rtg[b, :cur].unsqueeze(0),
                tstep[b, :cur].unsqueeze(0),
                attn,
            )  # -> (1,cur,A)

            preds[b, t] = out[:, -1]  # prediction for position t

    # zero pads; make contiguous defensively
    preds = (preds * masks.unsqueeze(-1)).contiguous()

    # --- build t+1 actions flattened in row-major valid order, no boolean indexing ---
    nxt_list = []
    for b in range(B):
        valid_len = int(masks[b].sum().item())
        for t in range(valid_len):
            if t + 1 < valid_len:
                nxt_list.append(preds[b, t + 1])
            else:
                nxt_list.append(torch.zeros(A, device=dev, dtype=acts.dtype))
    next_actions = torch.stack(nxt_list, dim=0).contiguous()  # (N_valid, A)

    return preds, next_actions

# @torch.no_grad()
# def predict_actions_with_trajectory_batch(
#     algo: TransformerAlgoImplBase,
#     trajectory_batch: TorchTrajectoryMiniBatch,    
#     *, 
#     mask_invalid_outputs: bool = True,
#     use_device: str | None = None,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Batched DT inference with strict causality (no future leakage).

#     Inputs
#     ------
#     algo: DecisionTransformer
#         Your algorithm instance. Must have `.impl._modules.transformer`.
#     traj_batch: TorchTrajectoryMiniBatch
#         Contains observations (B,L,...), actions (B,L,A), rewards (B,L,1),
#         returns_to_go (B,L,1), timesteps (B,L,1 or B,L), masks (B,L).
#         `masks==1` indicates valid steps; `0` is padding.
#     mask_invalid_outputs: bool
#         If True, zero out predictions at padded steps.
#     use_device: str | None
#         Optionally move tensors & model to a specific device.

#     Returns
#     -------
#     pred_actions: torch.Tensor
#         (B, L, A) DT outputs, one per timestep t, each using only ≤ t context.
#     pred_actions_flat: torch.Tensor
#         (N_valid, A) flattened predictions aligned to valid positions (useful
#         if you convert to a transition batch).
#     """

#     impl = algo.impl
#     transformer = impl._modules.transformer  # same module used in compute_loss
#     device = use_device or impl.device

#     # Move batch to device if needed
#     obs   = trajectory_batch.observations.to(device)
#     acts  = trajectory_batch.actions.to(device)
#     rtg   = trajectory_batch.returns_to_go.to(device)
#     tstep = trajectory_batch.timesteps.to(device)
#     masks = trajectory_batch.masks.to(device)              # (B, L), 1=valid, 0=pad

#     # ---- Strict autoregressive conditioning ----
#     # Predict a_t conditioned on { (s_0,a_0,r_0,rtg_0), ..., (s_{t-1},a_{t-1},r_{t-1},rtg_{t-1}), (s_t, rtg_t) }.
#     # Implement via teacher-forcing style right-shift of the action tokens:
#     #
#     #   actions_in[:, t] = ground-truth a_{t-1}
#     #   actions_in[:, 0] = 0 (pad action at the first step)
#     #
#     # This matches the online .inner_predict() pattern that appends a pad action.
#     actions_in = torch.zeros_like(acts)
#     actions_in[:, 1:] = acts[:, :-1]

#     # Attention mask convention in your code:
#     # inner_predict/compute_loss pass (1 - masks) to the transformer.
#     # Keep exactly the same here for consistency with training.
#     attention_mask = 1 - masks  # (B, L), 1=pad, 0=keep

#     # Forward pass: (B, L, A)
#     # Your transformer signature is:
#     #   transformer(observations, actions, returns_to_go, timesteps, attention_mask)
#     pred_actions = transformer(
#         obs,
#         actions_in,
#         rtg,
#         tstep,
#         attention_mask,
#     )

#     # Optionally zero predictions at padded positions for convenience
#     if mask_invalid_outputs:
#         pred_actions = pred_actions * masks.unsqueeze(-1)

#     # A flattened view aligned with valid timesteps (useful for transition targets)
#     valid_flat = masks.bool().view(-1)
#     pred_actions_flat = pred_actions.view(-1, pred_actions.size(-1))[valid_flat]

#     return pred_actions, pred_actions_flat


@dataclasses.dataclass(frozen=True)
class FQEBaseModules(Modules):
    q_funcs: nn.ModuleList
    targ_q_funcs: nn.ModuleList
    optim: OptimizerWrapper


class FQEBaseImpl(QLearningAlgoImplBase):
    _algo: QLearningAlgoImplBase | TransformerAlgoImplBase
    _modules: FQEBaseModules
    _gamma: float
    _q_func_forwarder: Union[
        DiscreteEnsembleQFunctionForwarder, ContinuousEnsembleQFunctionForwarder
    ]
    _targ_q_func_forwarder: Union[
        DiscreteEnsembleQFunctionForwarder, ContinuousEnsembleQFunctionForwarder
    ]
    _target_update_interval: int

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        algo: QLearningAlgoImplBase | TransformerAlgoImplBase,
        modules: FQEBaseModules,
        q_func_forwarder: Union[
            DiscreteEnsembleQFunctionForwarder,
            ContinuousEnsembleQFunctionForwarder,
        ],
        targ_q_func_forwarder: Union[
            DiscreteEnsembleQFunctionForwarder,
            ContinuousEnsembleQFunctionForwarder,
        ],
        gamma: float,
        target_update_interval: int,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._algo = algo
        self._gamma = gamma
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._target_update_interval = target_update_interval
        hard_sync(modules.targ_q_funcs, modules.q_funcs)

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        return self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    def compute_target(
        self, batch: TorchMiniBatch, next_actions: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations, next_actions
            )

    def update_target(self) -> None:
        hard_sync(self._modules.targ_q_funcs, self._modules.q_funcs)

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> dict[str, float]:
        next_actions = self._algo.predict_best_action(batch.next_observations)

        q_tpn = self.compute_target(batch, next_actions)
        loss = self.compute_loss(batch, q_tpn)

        self._modules.optim.zero_grad()
        loss.backward()
        self._modules.optim.step()

        if grad_step % self._target_update_interval == 0:
            self.update_target()

        return {"loss": float(loss.cpu().detach().numpy())}


class FQEImpl(ContinuousQFunctionMixin, FQEBaseImpl):
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder


class FQETrajectoryBaseImpl(QLearningAlgoImplBase):
    _algo: QLearningAlgoImplBase | TransformerAlgoImplBase
    _modules: FQEBaseModules
    _gamma: float
    _q_func_forwarder: Union[
        DiscreteEnsembleQFunctionForwarder, ContinuousEnsembleQFunctionForwarder
    ]
    _targ_q_func_forwarder: Union[
        DiscreteEnsembleQFunctionForwarder, ContinuousEnsembleQFunctionForwarder
    ]
    _target_update_interval: int
    _target_return: Optional[float]

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        algo: QLearningAlgoImplBase | TransformerAlgoImplBase,
        modules: FQEBaseModules,
        q_func_forwarder: Union[
            DiscreteEnsembleQFunctionForwarder,
            ContinuousEnsembleQFunctionForwarder,
        ],
        targ_q_func_forwarder: Union[
            DiscreteEnsembleQFunctionForwarder,
            ContinuousEnsembleQFunctionForwarder,
        ],
        gamma: float,
        target_update_interval: int,
        device: str,
        target_return: Optional[float] = None,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._algo = algo
        self._gamma = gamma
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._target_update_interval = target_update_interval
        self._target_return = target_return
        hard_sync(modules.targ_q_funcs, modules.q_funcs)

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        return self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    def compute_target(
        self, batch: TorchMiniBatch, next_actions: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations, next_actions
            )

    def update_target(self) -> None:
        hard_sync(self._modules.targ_q_funcs, self._modules.q_funcs)

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inner_update(self, batch: Union[TorchMiniBatch, TorchTrajectoryMiniBatch], grad_step: int) -> dict[str, float]:
        if isinstance(batch, TorchMiniBatch):
            # Default FQE behavior
            next_actions = self._algo.predict_best_action(batch.next_observations)
            q_tpn = self.compute_target(batch, next_actions)
            loss = self.compute_loss(batch, q_tpn)
        
        elif isinstance(batch, TorchTrajectoryMiniBatch):
            # Custom: generate predictions per timestep
            # NOTE: This assumes your Q-model can handle (B, L, ...) shape!

            ### TODO get next actions with a function like this
            
            preds, next_actions = dt_predict_next_actions(
                self._algo, batch, target_rtg=self._target_return
            )

            torch_transition_batch, _ = batch.to_transition_batch() # also outputs masks

            q_tpn = self.compute_target(torch_transition_batch, next_actions)
            loss = self.compute_loss(torch_transition_batch, q_tpn)
        
        
        self._modules.optim.zero_grad()
        loss.backward()
        self._modules.optim.step()

        if grad_step % self._target_update_interval == 0:
            self.update_target()

        return {"loss": float(loss.cpu().detach().numpy())}


class FQETrajectoryImpl(ContinuousQFunctionMixin, FQETrajectoryBaseImpl):
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder


class DiscreteFQETrajectoryImpl(DiscreteQFunctionMixin, FQETrajectoryBaseImpl):
    _q_func_forwarder: ContinuousEnsembleQFunctionForwarder
    _targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder

    def inner_update(self, batch: Union[TorchMiniBatch, TorchTrajectoryMiniBatch], grad_step: int) -> dict[str, float]:
        if isinstance(batch, TorchMiniBatch):
            # Default FQE behavior
            next_actions = self._algo.predict_best_action(batch.next_observations)
            q_tpn = self.compute_target(batch, next_actions)
            loss = self.compute_loss(batch, q_tpn)

        elif isinstance(batch, TorchTrajectoryMiniBatch):
            # Trajectory: query DT's action probabilities for expected-SARSA bootstrap
            _, next_action_probs = dt_predict_next_action_probs(
                self._algo, batch, target_rtg=self._target_return
            )

            torch_transition_batch, _ = batch.to_transition_batch()

            # Expected Q under policy: Σ_a' π(a'|s') · Q(s', a')
            B = next_action_probs.shape[0]
            next_obs = torch_transition_batch.next_observations
            q_next_all = self._targ_q_func_forwarder.compute_target(
                next_obs, torch.arange(self.action_size, device=self.device)
            )  # (B, A)
            expected_q = (next_action_probs * q_next_all).sum(dim=-1, keepdim=True)  # (B, 1)

            q_tpn = torch_transition_batch.rewards + self._gamma**torch_transition_batch.intervals * expected_q

            loss = self.compute_loss(torch_transition_batch, q_tpn)

        self._modules.optim.zero_grad()
        loss.backward()
        self._modules.optim.step()

        if grad_step % self._target_update_interval == 0:
            self.update_target()

        return {"loss": float(loss.cpu().detach().numpy())}



class DiscreteFQEImpl(DiscreteQFunctionMixin, FQEBaseImpl):
    _q_func_forwarder: ContinuousEnsembleQFunctionForwarder
    _targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        return self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    def compute_target(
        self, batch: TorchMiniBatch, next_actions: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                next_actions.long(),
            )

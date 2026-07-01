import dataclasses
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


from ....models.torch import (
    ContinuousDecisionTransformer,
    ContinuousEnsembleQFunctionForwarder,
    DiscreteDecisionTransformer,
    DiscreteEnsembleQFunctionForwarder,
)
from ....optimizers import OptimizerWrapper
from ....torch_utility import (
    CudaGraphWrapper,
    Modules,
    TorchMiniBatch,
    TorchTrajectoryMiniBatch,
    flatten_left_recursively,
    soft_sync,
)
from ....types import Shape
from ..base import TransformerAlgoImplBase
from ..inputs import TorchTransformerInput

__all__ = [
    "TACRImpl",
    "TACRModules",
    "DiscreteTACRModules",
    "DiscreteTACRImpl",
]


@dataclasses.dataclass(frozen=True)
class TACRModules(Modules):
    transformer: ContinuousDecisionTransformer
    actor_optim: OptimizerWrapper
    q_funcs: nn.ModuleList
    targ_q_funcs: nn.ModuleList
    critic_optim: OptimizerWrapper


class TACRImpl(TransformerAlgoImplBase):
    _modules: TACRModules
    _compute_actor_grad: Callable[[TorchTrajectoryMiniBatch], torch.Tensor]
    _compute_critic_grad: Callable[[TorchTrajectoryMiniBatch], torch.Tensor]

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: Modules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        alpha: float,
        gamma: float,
        tau: float,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        compiled: bool,
        device: str,
    ):
        super().__init__(observation_shape, action_size, modules, device)
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._alpha = alpha
        self._gamma = gamma
        self._tau = tau
        self._target_smoothing_sigma = target_smoothing_sigma
        self._target_smoothing_clip = target_smoothing_clip
        self._compute_actor_grad = (
            CudaGraphWrapper(self.compute_actor_grad)
            if compiled
            else self.compute_actor_grad
        )
        self._compute_critic_grad = (
            CudaGraphWrapper(self.compute_critic_grad)
            if compiled
            else self.compute_critic_grad
        )

    def inner_predict(self, inpt: TorchTransformerInput) -> torch.Tensor:
        # (1, T, A)
        action = self._modules.transformer(
            inpt.observations,
            inpt.actions,
            inpt.returns_to_go,
            inpt.timesteps,
            1 - inpt.masks,
        )
        # (1, T, A) -> (A,)
        return action[0][-1]

    def compute_actor_grad(
        self, batch: TorchTrajectoryMiniBatch
    ) -> torch.Tensor:
        self._modules.actor_optim.zero_grad()
        loss = self.compute_actor_loss(batch)
        loss.backward()
        return loss

    def compute_critic_grad(
        self, batch: TorchTrajectoryMiniBatch
    ) -> torch.Tensor:
        self._modules.critic_optim.zero_grad()
        transition_batch, masks = batch.to_transition_batch()
        q_tpn = self.compute_target(batch, transition_batch)
        loss = self.compute_critic_loss(transition_batch, q_tpn, masks)
        loss.backward()
        return loss

    def inner_update(
        self, batch: TorchTrajectoryMiniBatch, grad_step: int
    ) -> dict[str, float]:
        metrics = {}

        actor_loss = self._compute_actor_grad(batch)
        self._modules.actor_optim.step()
        metrics.update({"actor_loss": float(actor_loss.cpu().detach())})

        critic_loss = self._compute_critic_grad(batch)
        self._modules.critic_optim.step()
        soft_sync(self._modules.targ_q_funcs, self._modules.q_funcs, self._tau)
        metrics.update({"critic_loss": float(critic_loss.cpu().detach())})

        return metrics

    def compute_actor_loss(
        self, batch: TorchTrajectoryMiniBatch
    ) -> torch.Tensor:
        # (B, T, A)
        action = self._modules.transformer(
            batch.observations,
            batch.actions,
            batch.returns_to_go,
            batch.timesteps,
            1 - batch.masks,
        )
        # (B * T , 1)
        q_values = self._q_func_forwarder.compute_expected_q(
            x=flatten_left_recursively(batch.observations, dim=1),
            action=action.view(-1, self._action_size),
            reduction="min",
        )
        lam = self._alpha / q_values.abs().mean().clamp(min=1e-8).detach()
        q_loss = lam * -q_values
        # (B, T, A) -> (B, T)
        bc_loss = ((action - batch.actions) ** 2).sum(dim=-1)
        return (
            batch.masks.view(-1) * (q_loss.view(-1) + bc_loss.view(-1))
        ).mean()

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma,
            masks=masks,
        )
        return loss

    def compute_target(
        self, batch: TorchTrajectoryMiniBatch, transition_batch: TorchMiniBatch
    ) -> torch.Tensor:
        with torch.no_grad():
            # (B, T, A) -> (B * (T - 1), A)
            action = self._modules.transformer(
                batch.observations,
                batch.actions,
                batch.returns_to_go,
                batch.timesteps,
                1 - batch.masks,
            )[:, :-1].reshape(-1, self._action_size)
            # smoothing target
            noise = torch.randn(action.shape, device=batch.device)
            scaled_noise = self._target_smoothing_sigma * noise
            clipped_noise = scaled_noise.clamp(
                -self._target_smoothing_clip, self._target_smoothing_clip
            )
            smoothed_action = action + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
            return self._targ_q_func_forwarder.compute_target(
                transition_batch.next_observations,
                clipped_action,
                reduction="min",
            )


@dataclasses.dataclass(frozen=True)
class DiscreteTACRModules(Modules):
    transformer: DiscreteDecisionTransformer
    actor_optim: OptimizerWrapper
    q_funcs: torch.nn.ModuleList
    targ_q_funcs: torch.nn.ModuleList
    critic_optim: OptimizerWrapper


class DiscreteTACRImpl(TransformerAlgoImplBase):
    _modules: DiscreteTACRModules
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _compute_actor_grad: Callable[[TorchTrajectoryMiniBatch], torch.Tensor]
    _compute_critic_grad: Callable[[TorchTrajectoryMiniBatch], torch.Tensor]

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiscreteTACRModules,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        alpha: float,
        gamma: float,
        tau: float,
        compiled: bool,
        device: str,
    ):
        super().__init__(observation_shape, action_size, modules, device)
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._alpha = alpha
        self._gamma = gamma
        self._tau = tau
        # Setup JIT/CUDA-Graph Wrapping
        self._compute_actor_grad = (
            CudaGraphWrapper(self.compute_actor_grad) if compiled else self.compute_actor_grad
        )
        self._compute_critic_grad = (
            CudaGraphWrapper(self.compute_critic_grad) if compiled else self.compute_critic_grad
        )

    def inner_predict(self, inpt: TorchTransformerInput) -> torch.Tensor:
        # Logits statt kontinuierlicher Aktion
        _, logits = self._modules.transformer(
            inpt.observations,
            inpt.actions,
            inpt.returns_to_go,
            inpt.timesteps,
            1 - inpt.masks,
        )
        return logits[0, -1]

    def compute_actor_grad(self, batch: TorchTrajectoryMiniBatch) -> torch.Tensor:
        self._modules.actor_optim.zero_grad()
        loss = self.compute_actor_loss(batch)
        loss.backward()
        return loss

    def compute_critic_grad(self, batch: TorchTrajectoryMiniBatch) -> torch.Tensor:
        self._modules.critic_optim.zero_grad()
        transition_batch, masks = batch.to_transition_batch()
        q_target = self.compute_target(batch, transition_batch)
        loss = self.compute_critic_loss(transition_batch, q_target, masks)
        loss.backward()
        return loss

    def inner_update(
        self, batch: TorchTrajectoryMiniBatch, grad_step: int
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        actor_loss = self._compute_actor_grad(batch)
        self._modules.actor_optim.step()
        metrics["actor_loss"] = float(actor_loss.cpu().detach())

        critic_loss = self._compute_critic_grad(batch)
        self._modules.critic_optim.step()
        soft_sync(self._modules.targ_q_funcs, self._modules.q_funcs, self._tau)
        metrics["critic_loss"] = float(critic_loss.cpu().detach())

        return metrics

    def compute_actor_loss(self, batch: TorchTrajectoryMiniBatch) -> torch.Tensor:
        # (B, T, A)
        _, logits = self._modules.transformer(
            batch.observations,
            batch.actions,
            batch.returns_to_go,
            batch.timesteps,
            1 - batch.masks,
        )
        B, T, A = logits.shape
        logits_flat = logits.view(-1, A)  # (B*T, A)

        # Policy-Wahrscheinlichkeiten
        probs = F.softmax(logits_flat, dim=-1)  # (B*T, A)

        # Q(s,a) für alle Aktionen
        obs_flat = flatten_left_recursively(batch.observations, dim=1)
        q_all = self._q_func_forwarder.compute_expected_q(x=obs_flat)  # (B*T, A)

        # Erwartungswert unter der Policy
        expected_q = (probs * q_all).sum(dim=-1)  # (B*T,)

        # Gewichtung λ = α / E[|Q|]  — clamp to avoid div-by-zero early in training
        lam = self._alpha / expected_q.abs().mean().clamp(min=1e-8).detach()
        q_loss = lam * -expected_q  # (B*T,)

        # Behavior-Cloning-Loss (CE) gegen Expert-Aktionen
        bc_loss = F.cross_entropy(
            logits_flat,
            batch.actions.view(-1).long(),
            reduction="none",
        )  # (B*T,)

        mask = batch.masks.view(-1)  # (B*T,)
        return (mask * (q_loss + bc_loss)).mean()

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_target: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        # Standard Q-Learning-Fehler (MSE) über Ensemble
        actions_long = batch.actions.view(-1).long()
        actions_long = actions_long.view(batch.actions.shape)
        return self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=actions_long,
            rewards=batch.rewards,
            target=q_target,
            terminals=batch.terminals,
            gamma=self._gamma,
            masks=masks,
        )

    def compute_target(
        self,
        batch: TorchTrajectoryMiniBatch,
        transition_batch: TorchMiniBatch,
    ) -> torch.Tensor:
        with torch.no_grad():
            # Logits für t → t+1 (ohne letzten Zeitschritt)
            # _, logits = self._modules.transformer(
            #     batch.observations,
            #     batch.actions,
            #     batch.returns_to_go,
            #     batch.timesteps,
            #     1 - batch.masks,
            # )
            #next_logits = logits[:, :-1].reshape(-1, self._action_size)  # (B*(T-1), A)

            # Für diskrete Ziele: Max over actions
            q_next_all = self._targ_q_func_forwarder.compute_expected_q(
                x=transition_batch.next_observations
            )  # (B*(T-1), A)
            q_next = q_next_all.max(dim=-1)[0]  # (B*(T-1),)

            return q_next.unsqueeze(-1)
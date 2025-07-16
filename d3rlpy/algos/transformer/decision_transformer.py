import dataclasses

from ...base import DeviceArg, register_learnable
from ...constants import ActionSpace, PositionEncodingType
from ...models import EncoderFactory, make_encoder_field
from ...models.builders import (
    create_continuous_decision_transformer,
    create_discrete_decision_transformer,
)
from ...optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape,TorchObservation,NDArray, ObservationSequence
from .base import TransformerAlgoBase, TransformerConfig
from .torch.decision_transformer_impl import (
    DecisionTransformerImpl,
    DecisionTransformerModules,
    DiscreteDecisionTransformerImpl,
    DiscreteDecisionTransformerModules,
)

from ..qlearning import QLearningAlgoImplBase
from ...torch_utility import TorchMiniBatch
from .inputs import TorchTransformerInput, TransformerInput
import numpy as np
import torch
from types import MethodType
from typing import Sequence, Union, overload

__all__ = [
    "DecisionTransformerConfig",
    "DecisionTransformer",
    "DiscreteDecisionTransformerConfig",
    "DiscreteDecisionTransformer",
    "DTConstantRTGforFQE",
]


@dataclasses.dataclass()
class DecisionTransformerConfig(TransformerConfig):
    """Config of Decision Transformer.

    Decision Transformer solves decision-making problems as a sequence modeling
    problem.

    References:
        * `Chen at el., Decision Transformer: Reinforcement Learning via
          Sequence Modeling. <https://arxiv.org/abs/2106.01345>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        context_size (int): Prior sequence length.
        max_timestep (int): Maximum environmental timestep.
        batch_size (int): Mini-batch size.
        learning_rate (float): Learning rate.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of attention blocks.
        attn_dropout (float): Dropout probability for attentions.
        resid_dropout (float): Dropout probability for residual connection.
        embed_dropout (float): Dropout probability for embeddings.
        activation_type (str): Type of activation function.
        position_encoding_type (d3rlpy.PositionEncodingType):
            Type of positional encoding (``SIMPLE`` or ``GLOBAL``).
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    batch_size: int = 64
    learning_rate: float = 1e-4
    encoder_factory: EncoderFactory = make_encoder_field()
    optim_factory: OptimizerFactory = make_optimizer_field()
    num_heads: int = 1
    num_layers: int = 3
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    activation_type: str = "relu"
    position_encoding_type: PositionEncodingType = PositionEncodingType.SIMPLE
    compile_graph: bool = False

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "DecisionTransformer":
        return DecisionTransformer(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "decision_transformer"


class DecisionTransformer(
    TransformerAlgoBase[DecisionTransformerImpl, DecisionTransformerConfig]
):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        transformer = create_continuous_decision_transformer(
            observation_shape=observation_shape,
            action_size=action_size,
            encoder_factory=self._config.encoder_factory,
            num_heads=self._config.num_heads,
            max_timestep=self._config.max_timestep,
            num_layers=self._config.num_layers,
            context_size=self._config.context_size,
            attn_dropout=self._config.attn_dropout,
            resid_dropout=self._config.resid_dropout,
            embed_dropout=self._config.embed_dropout,
            activation_type=self._config.activation_type,
            position_encoding_type=self._config.position_encoding_type,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        optim = self._config.optim_factory.create(
            transformer.named_modules(),
            lr=self._config.learning_rate,
            compiled=self.compiled,
        )

        modules = DecisionTransformerModules(
            transformer=transformer,
            optim=optim,
        )

        self._impl = DecisionTransformerImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=self._device,
            compiled=self.compiled,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


@dataclasses.dataclass()
class DiscreteDecisionTransformerConfig(TransformerConfig):
    """Config of Decision Transformer for discrte action-space.

    Decision Transformer solves decision-making problems as a sequence modeling
    problem.

    References:
        * `Chen at el., Decision Transformer: Reinforcement Learning via
          Sequence Modeling. <https://arxiv.org/abs/2106.01345>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        context_size (int): Prior sequence length.
        max_timestep (int): Maximum environmental timestep.
        batch_size (int): Mini-batch size.
        learning_rate (float): Learning rate.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of attention blocks.
        attn_dropout (float): Dropout probability for attentions.
        resid_dropout (float): Dropout probability for residual connection.
        embed_dropout (float): Dropout probability for embeddings.
        activation_type (str): Type of activation function.
        embed_activation_type (str): Type of activation function applied to
            embeddings.
        position_encoding_type (d3rlpy.PositionEncodingType):
            Type of positional encoding (``SIMPLE`` or ``GLOBAL``).
        warmup_tokens (int): Number of tokens to warmup learning rate scheduler.
        final_tokens (int): Final number of tokens for learning rate scheduler.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    batch_size: int = 128
    learning_rate: float = 6e-4
    encoder_factory: EncoderFactory = make_encoder_field()
    optim_factory: OptimizerFactory = make_optimizer_field()
    num_heads: int = 8
    num_layers: int = 6
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    activation_type: str = "gelu"
    embed_activation_type: str = "tanh"
    position_encoding_type: PositionEncodingType = PositionEncodingType.GLOBAL
    warmup_tokens: int = 10240
    final_tokens: int = 30000000
    compile_graph: bool = False

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "DiscreteDecisionTransformer":
        return DiscreteDecisionTransformer(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "discrete_decision_transformer"


class DiscreteDecisionTransformer(
    TransformerAlgoBase[
        DiscreteDecisionTransformerImpl, DiscreteDecisionTransformerConfig
    ]
):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        transformer = create_discrete_decision_transformer(
            observation_shape=observation_shape,
            action_size=action_size,
            encoder_factory=self._config.encoder_factory,
            num_heads=self._config.num_heads,
            max_timestep=self._config.max_timestep,
            num_layers=self._config.num_layers,
            context_size=self._config.context_size,
            attn_dropout=self._config.attn_dropout,
            resid_dropout=self._config.resid_dropout,
            embed_dropout=self._config.embed_dropout,
            activation_type=self._config.activation_type,
            embed_activation_type=self._config.embed_activation_type,
            position_encoding_type=self._config.position_encoding_type,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        optim = self._config.optim_factory.create(
            transformer.named_modules(),
            lr=self._config.learning_rate,
            compiled=self.compiled,
        )

        modules = DiscreteDecisionTransformerModules(
            transformer=transformer,
            optim=optim,
        )

        self._impl = DiscreteDecisionTransformerImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            warmup_tokens=self._config.warmup_tokens,
            final_tokens=self._config.final_tokens,
            initial_learning_rate=self._config.learning_rate,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE





class DecisionTransformerImplforFQE(DecisionTransformerImpl):
    def __init__(self):
        pass

    def predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)
    
    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        batch_size = x.shape[0]
        return self._algo.predict(
            TransformerInput(
                observations=x,
                actions=np.zeros((batch_size, 1), dtype=np.float32),
                rewards=np.zeros((batch_size, 1), dtype=np.float32),
                returns_to_go=np.full((batch_size, 1), self._target_return, dtype=np.float32),
                timesteps=np.zeros((batch_size, 1), dtype=np.int64),
            )
        )

    def predict(self, x: TorchTransformerInput) -> torch.Tensor:
        return self.inner_predict(x)

    def sample_action(self, x: TorchObservation) -> torch.Tensor:
        raise NotImplementedError("Sampling is not supported in this wrapper.")

    def predict_value(self, x: TorchObservation, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Value prediction is not supported in this wrapper.")






# @overload
# def torch_to_numpy(obs: torch.Tensor) -> NDArray: ...
# @overload
# def torch_to_numpy(obs: Sequence[torch.Tensor]) -> Sequence[NDArray]: ...

# def torch_to_numpy(obs: TorchObservation) -> ObservationSequence:
#     """
#     Turn a TorchObservation (tensor *or* list/tuple of tensors)
#     into the matching ObservationSequence (ndarray *or* list/tuple
#     of ndarrays).

#     • Leaves the container structure (list vs tuple) unchanged.
#     • Ensures the array lives on CPU and is detached from the graph.
#     """
#     if isinstance(obs, torch.Tensor):
#         return obs.detach().cpu().numpy()
#     elif isinstance(obs, (list, tuple)):
#         convert = torch_to_numpy   # recursive alias
#         return type(obs)(convert(o) for o in obs)
#     else:                                  # defensive
#         raise TypeError(
#             f"Expected torch.Tensor or sequence thereof, got {type(obs)}"
#         )


# class DTConstantRTGforFQE(QLearningAlgoImplBase):
#     def __init__(self, dt_model: DecisionTransformer, target_return: float):
        
#         self._algo = dt_model
#         self.impl = dt_model.impl
#         self._target_return = target_return

#         super().__init__(  # ✔ let parent wire buffers
#             observation_shape=self.impl.observation_shape,
#             action_size=self.impl.action_size,
#             modules=self.impl.modules,
#             device=self.impl.device,
#         )

#         # ------------------------------------------------------------------
#         # Patch *this* impl instance only
#         # ------------------------------------------------------------------
#         wrapper = self  # capture in closure

#         def _inner(self_impl, x: TorchObservation) -> torch.Tensor:
#             x_np = torch_to_numpy(x)
#             if isinstance(x_np, (list, tuple)):
#                 batch = x_np[0].shape[0]
#             else:
#                 batch = x_np.shape[0]

#             return wrapper._algo.predict(
#                 TransformerInput(
#                     observations=x_np,
#                     actions=np.zeros((batch, 1), dtype=np.float32),
#                     rewards=np.zeros((batch, 1), dtype=np.float32),
#                     returns_to_go=np.full(
#                         (batch, 1), wrapper._target_return, dtype=np.float32
#                     ),
#                     timesteps=np.zeros((batch, 1), dtype=np.int64),
#                 )
#             )

#         # one‑liner that just delegates to inner
#         def _outer(self_impl, x: TorchObservation) -> torch.Tensor:
#             return self_impl.inner_predict_best_action(x)

#         # bind to *this* impl instance
#         self.impl.inner_predict_best_action = MethodType(_inner, self.impl)
#         self.impl.predict_best_action = MethodType(_outer, self.impl)

#     #     self._impl = dt_model._impl
#     #     self.impl = dt_model.impl

#     #     super().__init__(
#     #         observation_shape=self._impl.observation_shape,
#     #         action_size=self._impl.action_size,
#     #         modules=self._impl.modules,
#     #         device=self._impl.device,
#     #     )

#     #     self._algo = dt_model
#     #     self._target_return = target_return
#     #     self.impl.predict_best_action = MethodType(
#     #         self._predict_best_action, self.impl
#     #     )
#     #     self.impl.inner_predict_best_action = MethodType(
#     #         self._inner_predict_best_action, self.impl
#     #     )


#     # def _predict_best_action(self, x: TorchObservation) -> torch.Tensor:
#     #     return self._inner_predict_best_action(x)
    
#     # def _inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
#     #     x_np = torch_to_numpy(x)           # 👈 convert once, here
#     #     batch_size = x_np[0].shape[0] if isinstance(x_np, (list, tuple)) else x_np.shape[0]
#     #     return self._algo.predict(
#     #         TransformerInput(
#     #             observations=x_np,
#     #             actions=np.zeros((batch_size, 1), dtype=np.float32),
#     #             rewards=np.zeros((batch_size, 1), dtype=np.float32),
#     #             returns_to_go=np.full((batch_size, 1), self._target_return, dtype=np.float32),
#     #             timesteps=np.zeros((batch_size, 1), dtype=np.int64),
#     #         )
#     #     )

#     def _inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError("Sampling is not supported in this wrapper.")

#     def _inner_predict_value(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError("Value prediction is not supported in this wrapper.")

#     def _inner_update(self, batch: TorchMiniBatch, grad_step: int) -> dict[str, float]:
#         raise NotImplementedError("Updates are not supported in this wrapper.")

#     def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError("Sampling is not supported in this wrapper.")

#     def inner_predict_value(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError("Value prediction is not supported in this wrapper.")

#     def inner_update(self, batch: TorchMiniBatch, grad_step: int) -> dict[str, float]:
#         raise NotImplementedError("Updates are not supported in this wrapper.")
    
#     def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
#         return NotImplementedError("Best action prediction is not supported in this wrapper, done by impl.")


def torch_to_numpy(obs: TorchObservation) -> ObservationSequence:
    if isinstance(obs, torch.Tensor):
        return obs.detach().cpu().numpy()
    elif isinstance(obs, (list, tuple)):
        return type(obs)(torch_to_numpy(o) for o in obs)
    raise TypeError(f"Expected torch.Tensor or sequence, got {type(obs)}")


class DTConstantRTGforFQE(QLearningAlgoImplBase):
    def __init__(self, dt_model: DecisionTransformer, target_return: float):
        self._algo          = dt_model            # outer DecisionTransformer
        self.impl           = dt_model.impl       # inner model (weights)
        self._target_return = float(target_return)

        super().__init__(
            observation_shape=self.impl.observation_shape,
            action_size      =self.impl.action_size,
            modules          =self.impl.modules,
            device           =self.impl.device,
        )

        # ------------------------------------------------------------------
        # Patch just *this* impl instance
        # ------------------------------------------------------------------
        wrapper = self
        ctx     = dt_model._config.context_size    # typically 30

        def _inner(self_impl, x: TorchObservation) -> torch.Tensor:
            x_np  = torch_to_numpy(x)
            batch = x_np[0].shape[0] if isinstance(x_np, (list, tuple)) else x_np.shape[0]

            A = wrapper.impl.action_size  # <- number of action features

            action = wrapper._algo.predict(
                TransformerInput(
                    observations   = x_np,
                    actions        = np.zeros((batch, A), dtype=np.float32),
                    rewards        = np.zeros((batch, 1), dtype=np.float32),
                    returns_to_go  = np.full((batch, 1),
                                            wrapper._target_return, dtype=np.float32),
                    timesteps      = np.zeros(batch, dtype=np.int64),
                )
            )
            # --- NEW: ensure we hand a torch.Tensor back to FQE ------------------
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).to(wrapper.impl.device)
            return action

        def _outer(self_impl, x: TorchObservation) -> torch.Tensor:
            return self_impl.inner_predict_best_action(x)

        self.impl.inner_predict_best_action = MethodType(_inner,  self.impl)
        self.impl.predict_best_action       = MethodType(_outer, self.impl)

    # ----------------------------------------------------------------------
    # Minimal concrete implementations required by QLearningAlgoImplBase
    # ----------------------------------------------------------------------
    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return self.impl.inner_predict_best_action(x)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Sampling isn’t supported in Constant‑RTG FQE.")

    def inner_predict_value(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Value prediction isn’t supported in Constant‑RTG FQE.")

    def inner_update(self, batch: TorchMiniBatch, grad_step: int) -> dict[str, float]:
        raise NotImplementedError("Updates aren’t supported in Constant‑RTG FQE.")

register_learnable(DecisionTransformerConfig)
register_learnable(DiscreteDecisionTransformerConfig)


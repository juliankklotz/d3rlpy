import dataclasses

from ...base import DeviceArg, register_learnable
from ...constants import ActionSpace, PositionEncodingType
from ...models import EncoderFactory, make_encoder_field
from ...models.builders import (
    create_continuous_decision_transformer,
    create_continuous_q_function,
    create_discrete_decision_transformer,
    create_discrete_q_function,
)
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape
from .base import TransformerAlgoBase, TransformerConfig
from .torch.tacr_impl import (
    TACRImpl,
    TACRModules,
    DiscreteTACRImpl,
    DiscreteTACRModules,
)

__all__ = [
    "TACRConfig",
    "TACR",
    "DiscreteTACRConfig",
    "DiscreteTACR",
]


@dataclasses.dataclass()
class TACRConfig(TransformerConfig):
    """Config of Transformer Actor-Critic with Regularization.

    Decision Transformer-based actor-critic algorithm. The actor is modeled as
    Decision Transformer and additionally trained with a critic model. The
    extended actor-critic part is implemented as TD3+BC.

    References:
        * `Lee at el., Transformer Actor-Critic with Regularization: Automated
          Stock Trading using Reinforcement Learning.
          <https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p2815.pdf>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        context_size (int): Prior sequence length.
        max_timestep (int): Maximum environmental timestep.
        batch_size (int): Mini-batch size.
        actor_learning_rate (float): Learning rate for actor.
        critic_learning_rate (float): Learning rate for critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for critic.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for critic.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of attention blocks.
        attn_dropout (float): Dropout probability for attentions.
        resid_dropout (float): Dropout probability for residual connection.
        embed_dropout (float): Dropout probability for embeddings.
        activation_type (str): Type of activation function.
        position_encoding_type (d3rlpy.PositionEncodingType):
            Type of positional encoding (``SIMPLE`` or ``GLOBAL``).
        n_critics (int): Number of critics.
        alpha (float): Weight of Q-value actor loss.
        tau (float): Target network synchronization coefficiency.
        target_smoothing_sigma (float): Standard deviation for target noise.
        target_smoothing_clip (float): Clipping range for target noise.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    batch_size: int = 64
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    num_heads: int = 1
    num_layers: int = 3
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    activation_type: str = "relu"
    position_encoding_type: PositionEncodingType = PositionEncodingType.SIMPLE
    n_critics: int = 2
    alpha: float = 2.5
    tau: float = 0.005
    target_smoothing_sigma: float = 0.2
    target_smoothing_clip: float = 0.5
    compile_graph: bool = False

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "TACR":
        return TACR(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "tacr"


class TACR(TransformerAlgoBase[TACRImpl, TACRConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        transformer = create_continuous_decision_transformer(
            observation_shape=observation_shape,
            action_size=action_size,
            encoder_factory=self._config.actor_encoder_factory,
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
        optim = self._config.actor_optim_factory.create(
            transformer.named_modules(),
            lr=self._config.actor_learning_rate,
            compiled=self.compiled,
        )

        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(),
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )

        modules = TACRModules(
            transformer=transformer,
            actor_optim=optim,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            critic_optim=critic_optim,
        )

        self._impl = TACRImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            alpha=self._config.alpha,
            gamma=self._config.gamma,
            tau=self._config.tau,
            target_smoothing_sigma=self._config.target_smoothing_sigma,
            target_smoothing_clip=self._config.target_smoothing_clip,
            device=self._device,
            compiled=self.compiled,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


@dataclasses.dataclass()
class DiscreteTACRConfig(TransformerConfig):
    """Config of Transformer Actor-Critic with Regularization for discrete actions.

    Decision Transformer-based actor-critic algorithm for environments
    with discrete action spaces. The actor is modeled as a Decision Transformer
    and is additionally trained with a critic ensemble. The extended
    actor-critic part follows a discrete-action variant of TD3+BC.

    References:
        * Lee et al., Transformer Actor-Critic with Regularization:
          Automated Stock Trading using Reinforcement Learning.
          <https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p2815.pdf>_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler):
            Reward preprocessor.
        context_size (int):
            Number of past timesteps to condition on.
        max_timestep (int):
            Maximum environmental timestep for position encoding.
        batch_size (int):
            Mini-batch size.
        actor_learning_rate (float):
            Learning rate for the actor (Decision Transformer).
        critic_learning_rate (float):
            Learning rate for the critic ensemble.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor network.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for each critic network.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the critics.
        num_heads (int):
            Number of attention heads in the Transformer.
        num_layers (int):
            Number of Transformer encoder layers.
        attn_dropout (float):
            Dropout probability within attention layers.
        resid_dropout (float):
            Dropout probability on residual connections.
        embed_dropout (float):
            Dropout probability on input embeddings.
        activation_type (str):
            Activation function for Transformer blocks.
        embed_activation_type (str):
            Activation function for embedding projections.
        position_encoding_type (d3rlpy.PositionEncodingType):
            Type of positional encoding (“SIMPLE” or “GLOBAL”).
        n_critics (int):
            Number of critics in the ensemble.
        alpha (float):
            Weight of the Q-value term in the actor loss.
        tau (float):
            Soft-update coefficient for target critic networks.
        compile_graph (bool):
            Flag to enable JIT compilation and CUDAGraph for speed.
    """

    batch_size: int = 64
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4

    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    q_func_factory: QFunctionFactory = make_q_func_field()

    num_heads: int = 1
    num_layers: int = 3
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    activation_type: str = "relu"
    embed_activation_type: str = "tanh"
    position_encoding_type: PositionEncodingType = PositionEncodingType.SIMPLE

    # ensemble-parameter
    n_critics: int = 2

    # TACR-specific hyperparameters
    alpha: float = 2.5  # Gewicht des Q-Terms im Actor-Loss
    tau: float = 0.005  # Soft-Update-Koeffizient

    compile_graph: bool = False

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "DiscreteTACR":
        return DiscreteTACR(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "discrete_tacr"


class DiscreteTACR(
    TransformerAlgoBase[DiscreteTACRImpl, DiscreteTACRConfig]
):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        
        transformer = create_discrete_decision_transformer(
            observation_shape=observation_shape,
            action_size=action_size,
            encoder_factory=self._config.actor_encoder_factory,
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
        actor_optim = self._config.actor_optim_factory.create(
            transformer.named_modules(),
            lr=self._config.actor_learning_rate,
            compiled=self.compiled,
        )

        q_funcs, q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
            n_ensembles=self._config.n_critics,
        )
        targ_q_funcs, targ_q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
            n_ensembles=self._config.n_critics,
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(),
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )

        modules = DiscreteTACRModules(
            transformer=transformer,
            actor_optim=actor_optim,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            critic_optim=critic_optim,
        )

        self._impl = DiscreteTACRImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            alpha=self._config.alpha,
            gamma=self._config.gamma,
            tau=self._config.tau,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(TACRConfig)
register_learnable(DiscreteTACRConfig)
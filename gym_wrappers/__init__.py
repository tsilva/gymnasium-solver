from .CartPoleV1.reward_shaper import CartPoleV1_RewardShaper
from .discrete_encoder import DiscreteEncoder
from .env_wrapper_registry import EnvWrapperRegistry
from .MountainCarV0.reward_shaper import MountainCarV0_RewardShaper
from .pixel_observation import PixelObservationWrapper as _PixelObservationWrapper
from .discrete_action_space_remapper import (
    DiscreteActionSpaceRemapperWrapper,
)
from .action_reward_shaper import ActionRewardShaper
from .BreakoutV5.feature_extractor import BreakoutV5_FeatureExtractor
from .PongV5.feature_extractor import PongV5_FeatureExtractor
from .PongV5.reward_shaper import PongV5_RewardShaper
from .VizDoom.reward_shaper import VizDoom_RewardShaper

# VizDoom generic env is available via gym_wrappers.vizdoom: VizDoomEnv

class PixelObservationWrapper(_PixelObservationWrapper):
    """Alias wrapper to expose Gymnasium's PixelObservationWrapper via registry.

    Use with: { id: PixelObservationWrapper, pixels_only: true, render_kwargs: { } }
    """
    pass

_wrappers_to_register = [
    DiscreteEncoder,
    DiscreteActionSpaceRemapperWrapper,
    ActionRewardShaper,
    BreakoutV5_FeatureExtractor,
    PongV5_FeatureExtractor,
    PongV5_RewardShaper,
    MountainCarV0_RewardShaper,
    CartPoleV1_RewardShaper,
    VizDoom_RewardShaper,
    PixelObservationWrapper,
]
EnvWrapperRegistry.register(_wrappers_to_register)

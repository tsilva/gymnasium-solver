from .CartPoleV1.reward_shaper import CartPoleV1_RewardShaper
from .discrete_encoder import DiscreteEncoder
from .env_wrapper_registry import EnvWrapperRegistry
from .MountainCarV0.reward_shaper import MountainCarV0_RewardShaper
from .MountainCarV0.state_count_bonus import MountainCarV0_StateCountBonus
from .pixel_observation import PixelObservationWrapper as _PixelObservationWrapper
from .discrete_action_space_remapper import (
    DiscreteActionSpaceRemapperWrapper,
)
from .action_reward_shaper import ActionRewardShaper
from .sticky_actions import StickyActionsWrapper
from .BreakoutV5.feature_extractor import BreakoutV5_FeatureExtractor
from .PongV5.feature_extractor import PongV5_FeatureExtractor
from .PongV5.reward_shaper import PongV5_RewardShaper
from .VizDoom.reward_shaper import VizDoom_RewardShaper
from .RetroSuperMarioBros.reward_shaper import RetroSuperMarioBros_RewardShaper
from .frame_skip import FrameSkipWrapper
from .ale_action_masking import ALEActionMaskingWrapper

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
    StickyActionsWrapper,
    FrameSkipWrapper,
    BreakoutV5_FeatureExtractor,
    PongV5_FeatureExtractor,
    PongV5_RewardShaper,
    MountainCarV0_RewardShaper,
    MountainCarV0_StateCountBonus,
    CartPoleV1_RewardShaper,
    VizDoom_RewardShaper,
    RetroSuperMarioBros_RewardShaper,
    PixelObservationWrapper,
    ALEActionMaskingWrapper,
]
EnvWrapperRegistry.register(_wrappers_to_register)

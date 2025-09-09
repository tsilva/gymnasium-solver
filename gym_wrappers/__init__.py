from .discrete_to_binary import DiscreteToBinary
from .discrete_to_array import DiscreteToArray
from .env_wrapper_registry import EnvWrapperRegistry
from .MountainCarV0.reward_shaper import MountainCarV0_RewardShaper
from .PongV5.feature_extractor import PongV5_FeatureExtractor
from .PongV5.reward_shaper import PongV5_RewardShaper
from .CartPoleV1.reward_shaper import CartPoleV1_RewardShaper
from .VizDoom.reward_shaper import VizDoom_RewardShaper
from .pixel_observation import PixelObservationWrapper as _PixelObservationWrapper
# VizDoom generic env is available via gym_wrappers.vizdoom: VizDoomEnv

class PixelObservationWrapper(_PixelObservationWrapper):
    """Alias wrapper to expose Gymnasium's PixelObservationWrapper via registry.

    Use with: { id: PixelObservationWrapper, pixels_only: true, render_kwargs: { } }
    """
    pass

_wrappers_to_register = [
    DiscreteToBinary,
    DiscreteToArray,
    PongV5_FeatureExtractor,
    PongV5_RewardShaper,
    MountainCarV0_RewardShaper,
    CartPoleV1_RewardShaper,
    VizDoom_RewardShaper,
    PixelObservationWrapper,
]
EnvWrapperRegistry.register(_wrappers_to_register)
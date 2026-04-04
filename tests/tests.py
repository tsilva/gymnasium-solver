"""
Aggregator file for the curated `pytest tests/tests.py -v` workflow.

It re-exports the supported subset of tests that work as a single-file entrypoint
and gates optional-dependency suites explicitly.
"""

from importlib.util import find_spec

from tests.test_checkpoint import *  # noqa: F401,F403
from tests.test_config import *  # noqa: F401,F403
from tests.test_index_dataset import *  # noqa: F401,F403
from tests.test_logging_utils import *  # noqa: F401,F403
from tests.test_models import *  # noqa: F401,F403
from tests.test_multipass_random_sampler import *  # noqa: F401,F403
from tests.test_run_manager import *  # noqa: F401,F403

HAS_GYMNASIUM = find_spec("gymnasium") is not None

if HAS_GYMNASIUM:
    from tests.test_base_agent_helpers import *  # noqa: F401,F403
    from tests.test_env_wrapper_registry import *  # noqa: F401,F403
    from tests.test_mc_baseline_mask import *  # noqa: F401,F403
    from tests.test_ppo import *  # noqa: F401,F403
    from tests.test_ppo_integration import *  # noqa: F401,F403
    from tests.test_rollout_buffer import *  # noqa: F401,F403
    from tests.test_rollout_collector import *  # noqa: F401,F403
    from tests.test_rollouts import *  # noqa: F401,F403
    from tests.test_rollouts_extra import *  # noqa: F401,F403

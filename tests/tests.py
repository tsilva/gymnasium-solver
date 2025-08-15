"""
Aggregator file so the existing task `pytest tests/tests.py -v` discovers the full suite.

This file re-exports tests from the per-object test modules so pytest collects them
even when a single file is specified on the CLI.
"""

# Import and re-export all tests from the dedicated suites.
# Using absolute imports so it works without tests/__init__.py
from tests.test_index_dataset import *  # noqa: F401,F403
from tests.test_multipass_random_sampler import *  # noqa: F401,F403
from tests.test_base_agent_helpers import *  # noqa: F401,F403
from tests.test_config import *  # noqa: F401,F403
from tests.test_checkpoint import *  # noqa: F401,F403
from tests.test_run_manager import *  # noqa: F401,F403
from tests.test_logging_utils import *  # noqa: F401,F403
from tests.test_env_wrapper_registry import *  # noqa: F401,F403
from tests.test_models import *  # noqa: F401,F403
from tests.test_rollouts import *  # noqa: F401,F403


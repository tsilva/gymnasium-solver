from gym_wrappers.env_wrapper_registry import EnvWrapperRegistry


def is_alepy_env_id(env_id: str) -> bool:
    return env_id.startswith("ALE/")

def is_vizdoom_env_id(env_id: str) -> bool:
    # Support simple custom ids for VizDoom scenarios
    # Users can specify 'VizDoom-DeadlyCorridor-v0' or 'VizDoom-Basic-v0' in env_id
    return env_id in {
        "VizDoom-DeadlyCorridor-v0",
        "vizdoom-deadly-corridor",
        "VizDoom-Basic-v0",
        "vizdoom-basic",
        "VizDoom-DefendTheCenter-v0",
        "vizdoom-defend-the-center",
        "VizDoom-DefendTheLine-v0",
        "vizdoom-defend-the-line",
        "VizDoom-HealthGathering-v0",
        "vizdoom-health-gathering",
    }

def is_stable_retro_env_id(env_id: str) -> bool:
    """Heuristics to detect stable-retro (Gym Retro) environments.

    Supported patterns:
    - Prefix form: 'Retro/<GameName>' (recommended)
    - Direct game id for common titles (e.g., 'SuperMarioBros-Nes')
    """
    low = str(env_id).lower()
    if low.startswith("retro/"):
        return True
    # Minimal built-ins so users can pass the game id directly
    return env_id in {"SuperMarioBros-Nes"}

def is_rgb_env(env):
    import numpy as np
    from gymnasium import spaces

    # In case the observation space is not a box then it's not RGB
    if not isinstance(env.observation_space, spaces.Box):
        return False

    # If the observation space is not a 3D array then it's not RGB
    if len(env.observation_space.shape) < 3:
        return False

    # If the observation space is not uint8 then it's not RGB
    is_uint8 = getattr(env.observation_space, 'dtype', None) == np.uint8
    if not is_uint8:
        return False
    
    # If the observation space is not 3 channels then it's not RGB
    n_channels = env.observation_space.shape[-1]
    if not n_channels == 3:
        return False
    
    # Return True if all checks passed (is RGB)
    return True
    
def build_env(
    env_id, 
    n_envs=1, 
    seed=None, 
    max_episode_steps=None,
    env_wrappers=[], 
    grayscale_obs=False,
    resize_obs=False,
    norm_obs=False, 
    frame_stack=None, 
    obs_type=None,
    render_mode=None,
    subproc=None,
    record_video=False, 
    record_video_kwargs={},
    env_kwargs={}
):

    import gymnasium as gym
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecFrameStack,
        VecNormalize
    )

    from gym_wrappers.vec_info import VecInfoWrapper
    from gym_wrappers.vec_normalize_static import VecNormalizeStatic
    from gym_wrappers.vec_video_recorder import VecVideoRecorder
    
        
    # If recording video was requrested, assert valid render mode and subproc disabled 
    if record_video:
        if render_mode != "rgb_array": raise ValueError("Video recording requires render_mode='rgb_array'")
        if subproc: raise ValueError("Subprocess vector environments do not support video recording yet")

    _is_alepy = is_alepy_env_id(env_id)
    _is_vizdoom = is_vizdoom_env_id(env_id)
    _is_stable_retro = is_stable_retro_env_id(env_id)
    _is_pettingzoo_go = str(env_id).lower() in {"pettingzoo/go", "pettingzoo-go", "go"}

    # Default to RGB observations for ALE environments when obs_type is not specified.
    # Some callers (e.g., smoke tests) bypass full Config defaults and may pass None.
    if _is_alepy and obs_type is None:
        obs_type = "rgb"

    def env_fn():
        # In case this is an ale-based Atari environment
        if _is_alepy: 
            # Ensure ale_py is registered with gym (otherwise 
            # atari environments won't be available)
            import ale_py
            gym.register_envs(ale_py)
            
            # In case the observation type is objects, use OCAtari 
            # to extract object-based observations from game RAM
            if obs_type == "objects":
                from ocatari.core import OCAtari
                env = OCAtari(env_id, mode="ram", hud=False, render_mode=render_mode, **env_kwargs)
            # Otherwise, create the standard ALE environment
            else:
                env = gym.make(env_id, obs_type=obs_type, render_mode=render_mode, **env_kwargs)

                # Apply Gymnasium's Atari preprocessing for RGB observations.
                # This consolidates grayscale/resize/frameskip behavior and avoids
                # conflicts with manual wrappers configured elsewhere.
                applied_atari_preproc = False
                try:
                    if str(obs_type).lower() == "rgb":
                        try:
                            # Prefer modern import location
                            from gymnasium.wrappers.atari import AtariPreprocessing  # type: ignore
                        except Exception:
                            # Fallback for older gymnasium versions
                            from gymnasium.wrappers import AtariPreprocessing  # type: ignore

                        # Use common defaults (grayscale+resize to 84x84, frame_skip=4)
                        env = AtariPreprocessing(
                            env,
                            grayscale_obs=True,
                            scale_obs=False,
                            screen_size=84,
                            frame_skip=4,
                            terminal_on_life_loss=False,
                        )
                        applied_atari_preproc = True
                except Exception:
                    # Be robust: if wrapper import/application fails, proceed without it
                    applied_atari_preproc = False
        # VizDoom custom integrations
        elif _is_vizdoom:
            from gym_wrappers.vizdoom import VizDoomEnv
            env_id_lower = str(env_id).lower()
            scenario_map = {
                "vizdoom-deadly-corridor": "deadly_corridor",
                "vizdoom-deadlycorridor-v0": "deadly_corridor",
                "vizdoom-deadlycorridor": "deadly_corridor",
                "vizdoom-deadlycorridor-v1": "deadly_corridor",
                "vizdoom-deadly-corridor-v0": "deadly_corridor",
                "vizdoom-basic": "basic",
                "vizdoom-basic-v0": "basic",
                "vizdoom-defend-the-center": "defend_the_center",
                "vizdoom-defend-the-center-v0": "defend_the_center",
                "vizdoom-defend-the-line": "defend_the_line",
                "vizdoom-defend-the-line-v0": "defend_the_line",
                "vizdoom-health-gathering": "health_gathering",
                "vizdoom-health-gathering-v0": "health_gathering",
                # Canonical IDs
                "vizdoom-deadlycorridor-v0": "deadly_corridor",
            }
            # Normalize some canonical forms used by configs
            canonical = env_id_lower.replace("/", "-")
            if canonical.startswith("vizdoom-"):
                scenario = scenario_map.get(canonical)
            else:
                scenario = None
            if scenario is None:
                # Strict mapping for known IDs
                if env_id in {"VizDoom-DeadlyCorridor-v0", "vizdoom-deadly-corridor"}:
                    scenario = "deadly_corridor"
                elif env_id in {"VizDoom-Basic-v0", "vizdoom-basic"}:
                    scenario = "basic"
                elif env_id in {"VizDoom-DefendTheCenter-v0", "vizdoom-defend-the-center"}:
                    scenario = "defend_the_center"
                elif env_id in {"VizDoom-DefendTheLine-v0", "vizdoom-defend-the-line"}:
                    scenario = "defend_the_line"
                elif env_id in {"VizDoom-HealthGathering-v0", "vizdoom-health-gathering"}:
                    scenario = "health_gathering"
            if scenario is None:
                # Fallback to deadly corridor if unknown vizdoom id
                scenario = "deadly_corridor"
            env = VizDoomEnv(scenario=scenario, render_mode=render_mode, **env_kwargs)
        # Stable-Retro (Gym Retro) integrations
        elif _is_stable_retro:
            try:
                import retro  # type: ignore
            except Exception as e:
                raise ImportError(
                    "stable-retro is required for Retro environments. Install with `pip install stable-retro`"
                ) from e

            # Allow both 'Retro/<Game>' and bare game id (e.g., 'SuperMarioBros-Nes')
            if str(env_id).lower().startswith("retro/"):
                game = str(env_id).split("/", 1)[1]
            else:
                game = str(env_id)

            # Extract Retro-specific kwargs while keeping user overrides
            make_kwargs = dict(env_kwargs) if isinstance(env_kwargs, dict) else {}
            # Prefer a discrete action space so our categorical policies output
            # integer actions compatible with Retro's internal encoding.
            # Users can override via env_kwargs if they need MultiDiscrete/MultiBinary.
            make_kwargs.setdefault("use_restricted_actions", getattr(retro, "Actions").DISCRETE)
            # Support 'state' override via env_kwargs; None → retro's default state
            state = make_kwargs.pop("state", None)

            # stable-retro supports Gymnasium-style render_mode
            env = retro.make(game=game, state=state, render_mode=render_mode, **make_kwargs)
        # PettingZoo Go (wrapped single-agent)
        elif _is_pettingzoo_go:
            # Lazy import to avoid hard dependency for users who don't need it
            from pettingzoo.classic import go_v5
            from gym_wrappers.pettingzoo_single_agent import PettingZooSingleAgentWrapper

            # Extract supported kwargs
            board_size = int(env_kwargs.pop("board_size", 9) or 9)
            opponent = env_kwargs.pop("opponent", "random")  # placeholder for future extensions

            # Prefer parallel env if available, otherwise fall back to AEC env
            try:
                if hasattr(go_v5, "parallel_env"):
                    pz_env = go_v5.parallel_env(board_size=board_size, render_mode=render_mode, **env_kwargs)
                elif hasattr(go_v5, "env"):
                    pz_env = go_v5.env(board_size=board_size, render_mode=render_mode, **env_kwargs)
                else:
                    pz_env = go_v5.raw_env(board_size=board_size, render_mode=render_mode, **env_kwargs)  # type: ignore[attr-defined]
            except Exception:
                # Final fallback in case signature differs between versions
                if hasattr(go_v5, "env"):
                    pz_env = go_v5.env(board_size=board_size)
                else:
                    pz_env = go_v5.raw_env(board_size=board_size)  # type: ignore[attr-defined]

            # Control the first agent by default
            env = PettingZooSingleAgentWrapper(pz_env, agent_id=None, render_mode=render_mode)
        # Otherwise, create a standard gym environment
        else: 
            # Standard Gymnasium environments with version fallback handling
            try:
                env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
            except Exception as e:
                # Gracefully handle version mismatches (e.g., 'LunarLander-v3' on
                # installs that only provide 'v2'). We attempt to decrement the
                # version until a working one is found.
                try:
                    from gymnasium import error as gym_error  # type: ignore
                except Exception:
                    gym_error = None  # best-effort fallback below

                # Detect a version-not-found scenario
                is_version_issue = False
                if gym_error is not None and hasattr(gym_error, "VersionNotFound"):
                    is_version_issue = isinstance(e, gym_error.VersionNotFound)  # type: ignore[attr-defined]
                if not is_version_issue:
                    # Heuristic fallback based on message text
                    msg = str(e).lower()
                    is_version_issue = ("version" in msg and ("doesn't exist" in msg or "not found" in msg))

                if is_version_issue:
                    import re
                    env = None  # ensure defined for post-loop checks
                    last_exc = e
                    m = re.match(r"^(.*)-v(\d+)$", str(env_id))
                    if m:
                        base, ver_str = m.group(1), m.group(2)
                        try:
                            ver = int(ver_str)
                        except Exception:
                            ver = None
                        if isinstance(ver, int):
                            for new_ver in range(ver - 1, -1, -1):
                                fallback_id = f"{base}-v{new_ver}"
                                try:
                                    env = gym.make(fallback_id, render_mode=render_mode, **env_kwargs)
                                    try:
                                        # Inform user once per process; safe to ignore failures here
                                        print(f"[env] '{env_id}' not available; falling back to '{fallback_id}'.")
                                    except Exception:
                                        pass
                                    # Expose resolved id for downstream introspection if needed
                                    try:
                                        setattr(env, "resolved_env_id", fallback_id)
                                    except Exception:
                                        pass
                                    break
                                except Exception as exc:
                                    last_exc = exc
                                    env = None  # continue searching lower versions
                                    continue
                        # If no fallback succeeded, re-raise original error
                        if env is None:
                            # Prefer the last exception (e.g., missing Box2D) for clarity
                            raise last_exc
                    else:
                        # Not a versioned id (e.g., custom id), re-raise
                        raise
                else:
                    # Different kind of failure; propagate
                    raise

    # NOTE: Do not auto-wrap discrete observation spaces here to avoid
    # impacting tabular algorithms (e.g., Q-Learning) that rely on
    # Discrete observation IDs. Instead, VecInfoWrapper exposes an
    # input_dim for Discrete spaces (1), enabling MLP policies to work.
        
        # Important: avoid manual resize/grayscale for ALE RGB when AtariPreprocessing is applied
        # (it already handles grayscale+resize). For other envs, keep the knobs.
        if not (_is_alepy and str(obs_type).lower() == "rgb" and locals().get("applied_atari_preproc", False)):
            # Resize before grayscale to avoid cv2 dropping the channel dim on (H,W,1)
            if resize_obs:
                from gymnasium.wrappers import ResizeObservation
                env = ResizeObservation(env, shape=(84, 84))  # TODO: softcode this

            if grayscale_obs:
                from gymnasium.wrappers import GrayscaleObservation
                env = GrayscaleObservation(env, keep_dim=True)

        # Apply configured env wrappers
        for wrapper in env_wrappers:
            env = EnvWrapperRegistry.apply(env, wrapper) # type: ignore

        if max_episode_steps is not None:
            from gymnasium.wrappers import TimeLimit
            env = TimeLimit(env, max_episode_steps=max_episode_steps)

        # Return the environment
        return env

    # Vectorize the environment
    # Retro emulators cannot have multiple instances in the same process.
    # Use subprocess vectorization for multi-env Retro, or fall back to a single
    # env when video recording is enabled (which requires subproc=False).
    effective_n_envs = int(n_envs or 1)
    if _is_stable_retro and record_video and effective_n_envs > 1:
        # Keep user UX smooth: silently reduce to one env for recorded eval/test
        # runs where subproc must be disabled.
        effective_n_envs = 1

    vec_env_cls = DummyVecEnv
    if subproc is not None:
        vec_env_cls = SubprocVecEnv if subproc else DummyVecEnv
    else:
        # Prefer subprocesses when needed/beneficial
        if (_is_alepy or _is_stable_retro) and effective_n_envs > 1:
            vec_env_cls = SubprocVecEnv
        else:
            vec_env_cls = DummyVecEnv

    vec_env_kwargs = {"start_method": "spawn"} if vec_env_cls == SubprocVecEnv else {}
    env = make_vec_env(env_fn, n_envs=effective_n_envs, seed=seed, vec_env_cls=vec_env_cls, vec_env_kwargs=vec_env_kwargs)

    # Ensure the vectorized env exposes render_mode for downstream wrappers (e.g., video recorder)
    try:
        setattr(env, "render_mode", render_mode)
    except Exception:
        pass

    # Expose original env_id on the vectorized env for downstream introspection (e.g., YAML fallbacks)
    try:
        setattr(env, "env_id", env_id)
    except Exception:
        pass

    # Detect image observations and transpose to channel-first for SB3 CNN policies
    from stable_baselines3.common.preprocessing import is_image_space
    try:
        is_image = is_image_space(env.observation_space, check_channels=False)
    except Exception:
        is_image = False

    from stable_baselines3.common.vec_env import VecTransposeImage
    # Only transpose if observations are channel-last images (H, W, C). Grayscale
    # Atari after preprocessing is (H, W) and should not be transposed here.
    obs_shape = getattr(env.observation_space, "shape", tuple())
    if is_image and isinstance(obs_shape, tuple) and len(obs_shape) == 3:
        env = VecTransposeImage(env)  # (N, C, H, W)

    # Enable observation normalization only for non-image observations
    if not is_image:
        if norm_obs == "static":
            env = VecNormalizeStatic(env)
        elif norm_obs is True:
            env = VecNormalize(env, norm_obs=norm_obs)

    # Enable frame stacking if requested
    if frame_stack and frame_stack > 1: env = VecFrameStack(env, n_stack=frame_stack)
    
    # Enable video recording if requested
    # record_video_kwargs may include: video_length, record_env_idx (to record a single env)
    if record_video:
        env = VecVideoRecorder(
            env,
            **record_video_kwargs
        )

    # Add instrospection info wrapper that 
    # allows easily querying for env details
    # through the vectorized wrapper
    # This should be added last to get the final observation space dimensions
    env = VecInfoWrapper(env)

    return env

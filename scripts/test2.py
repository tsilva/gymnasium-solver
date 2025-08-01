from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from env_wrappers.vec_video_recorder import VecVideoRecorder
import gymnasium as gym
from env_wrappers.reward_shaper_mountaincar_v0 import MountainCarRewardShaping

def build_env(
    env_id,
    n_envs: int = 1,
    seed: int | None = None,
    vec_env_cls=None,            # may be class or string: "SubprocVecEnv"/"DummyVecEnv"
    env_wrappers=None,           # ordered list of per-env wrappers
    vec_wrappers=None,           # ordered list of vec wrappers
    env_make=None,               # callable or "pkg.mod:func" (defaults to gymnasium.make)
    make_kwargs=None,            # forwarded to env_make
):
    """
    Wrapper spec formats (applied IN ORDER):
      - WrapperClass
      - (WrapperClass, {"arg": value})
      - "package.module:ClassName"
      - {"cls": WrapperClass_or_str, "kwargs": {...}}

    Notes:
      - Put reward shaping, obs/action transforms, TimeLimit, etc. in env_wrappers.
      - Put VecNormalize, VecFrameStack, VecVideoRecorder, etc. in vec_wrappers.
      - If any wrapper needs frames, set render_mode via make_kwargs={"render_mode": "rgb_array"}.
      - For SubprocVecEnv, make sure classes/funcs are top-level (pickleable).
    """
    import importlib
    import gymnasium as gym
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    make_kwargs = dict(make_kwargs or {})
    env_wrappers = list(env_wrappers or [])
    vec_wrappers = list(vec_wrappers or [])

    # Resolve vec env class from string for convenience
    if isinstance(vec_env_cls, str):
        if vec_env_cls.lower() in {"subprocvecenv", "subproc"}:
            vec_env_cls = SubprocVecEnv
        elif vec_env_cls.lower() in {"dummyvecenv", "dummy"}:
            vec_env_cls = DummyVecEnv

    def _resolve_callable(spec):
        if spec is None:
            return None
        if callable(spec):
            return spec
        if isinstance(spec, str):
            mod, _, name = spec.partition(":")
            if not _:
                # also allow dotted "pkg.mod.func"
                mod, name = spec.rsplit(".", 1)
            return getattr(importlib.import_module(mod), name)
        raise TypeError(f"Cannot resolve callable from {spec!r}")

    def _resolve_cls(spec):
        if isinstance(spec, str):
            mod, _, name = spec.partition(":")
            if not _:
                mod, name = spec.rsplit(".", 1)
            return getattr(importlib.import_module(mod), name)
        return spec

    def _split_spec(item):
        # returns (cls, kwargs)
        if isinstance(item, tuple):
            cls, kwargs = item
            return _resolve_cls(cls), dict(kwargs or {})
        if isinstance(item, dict):
            cls = _resolve_cls(item.get("cls") or item.get("callable"))
            return cls, dict(item.get("kwargs") or {})
        return _resolve_cls(item), {}

    def _apply_wrappers(obj, wrappers):
        for w in wrappers:
            cls, kwargs = _split_spec(w)
            obj = cls(obj, **kwargs)
        return obj

    # Choose env make function (default: gymnasium.make)
    make_fn = _resolve_callable(env_make) or gym.make

    # Per-process env factory (safe for SubprocVecEnv if all specs are importable / top-level)
    def env_fn():
        env = make_fn(env_id, **make_kwargs)
        env = _apply_wrappers(env, env_wrappers)
        return env

    # Vectorize
    venv = make_vec_env(env_fn, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)

    # Apply vec-level wrappers
    venv = _apply_wrappers(venv, vec_wrappers)
    return venv

env = build_env(
    "MountainCar-v0",
    n_envs=8,
    vec_env_cls="DummyVecEnv",
    make_kwargs={"render_mode": "rgb_array"},   # needed for video
   # env_wrappers=[
   #     gym.wrappers.RecordEpisodeStatistics,
   #     (gym.wrappers.TimeLimit, {"max_episode_steps": 200})#,
   #     #(MountainCarRewardShaping, {"alpha": 0.1}),
   # ],
    vec_wrappers=[
        (VecNormalize, {"norm_obs": True, "norm_reward": True}),
        (VecFrameStack, {"n_stack": 4})#,
        #(VecVideoRecorder, {
        #    "video_folder": "videos",
        #    "record_video_trigger": "my_pkg.triggers:every_10k_steps",  # or a top-level callable
        #    "video_length": 1000
        #}),
    ],
)

env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    if done:
        obs, info = env.reset()
    env.render()  # Render the environment (if applicable)
env.close()
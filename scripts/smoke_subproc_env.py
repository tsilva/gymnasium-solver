import os


def main():
    from utils.environment import build_env

    env = build_env(
        env_id="ALE/Pong-v5",
        n_envs=2,
        seed=123,
        obs_type="objects",
        render_mode=None,
        subproc=True,
        env_wrappers=[{"id": "PongV5_FeatureExtractor"}],
    )

    # This calls into VecInfoWrapper methods
    env.print_spec()

    # Do a minimal reset/close cycle
    env.reset()
    env.close()
    print("OK: SubprocVecEnv smoke test passed")


if __name__ == "__main__":
    # Ensure cwd is repo root when executed directly
    repo_root = os.path.dirname(os.path.dirname(__file__))
    os.chdir(repo_root)
    main()



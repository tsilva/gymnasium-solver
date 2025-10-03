#!/usr/bin/env python3
"""Test script to verify stable-retro compatibility on M1 Mac."""

import sys
import traceback


def test_import():
    """Test if retro module can be imported."""
    print("Test 1: Importing retro module...")
    try:
        import retro
        print(f"✓ Successfully imported retro (version: {retro.__version__ if hasattr(retro, '__version__') else 'unknown'})")
        return True
    except Exception as e:
        print(f"✗ Failed to import retro: {e}")
        traceback.print_exc()
        return False


def test_list_games():
    """Test if retro can list available games."""
    print("\nTest 2: Listing available games...")
    try:
        import retro
        games = retro.data.list_games()
        print(f"✓ Successfully listed {len(games)} games")
        if games:
            print(f"  Sample games: {list(games)[:5]}")
        return True
    except Exception as e:
        print(f"✗ Failed to list games: {e}")
        traceback.print_exc()
        return False


def test_create_env():
    """Test if retro can create a basic environment."""
    print("\nTest 3: Creating test environment...")
    try:
        import retro
        games = retro.data.list_games()
        if not games:
            print("⚠ No games available, skipping environment creation test")
            return True

        test_game = list(games)[0]
        print(f"  Attempting to create environment for: {test_game}")
        env = retro.make(game=test_game, use_restricted_actions=retro.Actions.DISCRETE)
        print(f"✓ Successfully created environment for {test_game}")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        traceback.print_exc()
        return False


def test_env_step():
    """Test if retro environment can be stepped."""
    print("\nTest 4: Testing environment step...")
    try:
        import retro
        games = retro.data.list_games()
        if not games:
            print("⚠ No games available, skipping environment step test")
            return True

        test_game = list(games)[0]
        env = retro.make(game=test_game, use_restricted_actions=retro.Actions.DISCRETE)
        obs, info = env.reset()
        print(f"  Initial observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")

        # Take a few random steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

        print(f"✓ Successfully stepped through environment")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Failed to step environment: {e}")
        traceback.print_exc()
        return False


def test_gymnasium_integration():
    """Test if our environment.py helper works with stable-retro."""
    print("\nTest 5: Testing gymnasium-solver integration...")
    try:
        from utils.environment import _is_stable_retro_env_id, build_env

        # Test ID detection
        assert _is_stable_retro_env_id("Retro/SuperMarioBros-Nes")
        assert not _is_stable_retro_env_id("CartPole-v1")
        print("✓ Retro env ID detection works")

        # Try to build an environment
        import retro
        games = retro.data.list_games()
        if games:
            test_game = list(games)[0]
            env_id = f"Retro/{test_game}"
            print(f"  Attempting to build environment for: {env_id}")
            vec_env = build_env(env_id, n_envs=1, seed=42)
            print(f"✓ Successfully built vectorized environment")
            vec_env.close()
        else:
            print("⚠ No games available, skipping integration test")

        return True
    except Exception as e:
        print(f"✗ Failed integration test: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("stable-retro Compatibility Test on M1 Mac")
    print("=" * 60)

    tests = [
        test_import,
        test_list_games,
        test_create_env,
        test_env_step,
        test_gymnasium_integration,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("✓ All tests passed! stable-retro is working on M1 Mac")
        return 0
    else:
        print("✗ Some tests failed. stable-retro may have compatibility issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())

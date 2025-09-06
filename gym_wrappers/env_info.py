import os
import yaml
import gymnasium as gym
from dataclasses import asdict

class EnvInfoWrapper(gym.ObservationWrapper):

    def get_id(self):
        return self.env.spec.id

    def _get_spec__file(self):
        env_id = self.get_id()
        spec_path = f"config/environments/{env_id}.spec.yaml"
        if not os.path.exists(spec_path): return {}
        with open(spec_path, "r") as f: spec = yaml.safe_load(f)
        return spec

    def _get_spec__env(self):
        return asdict(self.env.spec)

    def get_spec(self):
        _file_spec = self._get_spec__file()
        _env_spec = self._get_spec__env()
        spec = {**_env_spec, **_file_spec}
        return spec

    def get_max_episode_steps(self):
        spec = self.get_spec()
        max_episode_steps = spec['max_episode_steps']
        return max_episode_steps

    def get_reward_treshold(self):
        spec = self.get_spec()
        reward_threshold = spec['reward_threshold']
        return reward_threshold

    # ObservationWrapper requires implementing this method. We do not
    # transform observations; return them unchanged.
    def observation(self, observation):  # type: ignore[override]
        return observation

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env_info = EnvInfoWrapper(env)
    print(env_info.get_spec())
    print(env_info.get_reward_treshold())

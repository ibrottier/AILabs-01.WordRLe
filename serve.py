import gym
from starlette.requests import Request
import requests
from main import WordleEnv
import ray
import ray.rllib.agents.ppo as ppo
from ray import serve

@serve.deployment(route_prefix="/test")
class ServePPOModel:
    def __init__(self, checkpoint_path) -> None:
        self.trainer = ppo.PPOTrainer(
            env="my_env",
        )
        self.trainer.restore(checkpoint_path)

    async def __call__(self, request: Request):
        json_input = await request.json()
        obs = json_input["observation"]

        action = self.trainer.compute_action(obs)
        return {"action": int(action)}


if __name__ == '__main__':
    from ray import tune
    tune.register_env("my_env", lambda config: WordleEnv())

    serve.start()
    ServePPOModel.deploy(r'C:\Users\patricio.ivan.pipp\ray_results\PPO_WordRLe - Word Driven v 1.2_2022-01-31_17-46-55focikk4j\checkpoint_000011\checkpoint-11')

    # That's it! Let's test it
    for _ in range(10):
        env = WordleEnv()
        obs = env.reset()

        print(f"-> Sending observation {obs}")
        obs_ = [o.tolist() for o in obs]
        print(obs_)

        resp = requests.get(
            "http://localhost:8000/test", json={"observation": obs_}
        )
        print(f"<- Received response {resp.json()}")

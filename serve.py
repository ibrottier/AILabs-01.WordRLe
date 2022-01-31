import gym
from starlette.requests import Request
import requests
from main import WordleEnv
import ray
import ray.rllib.agents.ppo as ppo
from ray import serve

def train_ppo_model():
    trainer = ppo.PPOTrainer(
        config={"framework": "torch", "num_workers": 0},
        env="CartPole-v0",
    )
    # Train for one iteration
    trainer.train()
    trainer.save("/tmp/rllib_checkpoint")
    return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"


checkpoint_path = train_ppo_model()

@serve.deployment
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
    ServePPOModel.deploy(r'C:\Users\patricio.ivan.pipp\ray_results\PPO_my_env_2022-01-28_17-53-32w2cu9v5s\checkpoint_000593\checkpoint-593')

    # That's it! Let's test it
    for _ in range(10):
        env = WordleEnv()
        obs = env.reset()

        print(f"-> Sending observation {obs}")
        obs_ = [obs.tolist() for o in obs]
        print(obs_)

        resp = requests.get(
            "http://localhost:8000", json={"observation": obs.tolist()}
        )
        print(f"<- Received response {resp.json()}")
    # Output:
    # <- Received response {'action': 1}
    # -> Sending observation [0.04228249 0.02289503 0.00690076 0.03095441]
    # <- Received response {'action': 0}
    # -> Sending observation [ 0.04819471 -0.04702759 -0.00477937 -0.00735569]
    # <- Received response {'action': 0}
    # ...
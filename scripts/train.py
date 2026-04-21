import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.common.wrappers import ActionMasker

from envs.connect4_wrapper import OneAgentVsRandomGym

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Connect4 agent with PPO or MaskablePPO."
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "maskable_ppo"],
        default="ppo",
        help="Algorithm to use.",
    )
    return parser.parse_args()


def mask_fn(env) -> np.ndarray:
    return env.action_masks()


def make_env(eval_seed: int, algo: str):
    def _factory():
        env = OneAgentVsRandomGym(seed=eval_seed)
        if algo == "maskable_ppo":
            env = ActionMasker(env, mask_fn)
        return env
    return _factory


def create_model(algo: str, seed:int, vec_env: DummyVecEnv):
    # --- MODEL ---
    modelCls = MaskablePPO if algo == "maskable_ppo" else PPO

    model = modelCls(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
        tensorboard_log="runs",
        seed=seed,
        verbose=1,
    )

    return model


def main():
    args = parse_args()

    for seed in range(0,3):
        env_fns = [make_env(seed + i*10, args.algo) for i in range(4)]
        vec_env = DummyVecEnv(env_fns)

        model = create_model(algo=args.algo, vec_env=vec_env, seed=seed)

        checkpoint_callback = CheckpointCallback(
            save_freq=64,
            save_path="checkpoints",
            name_prefix=f"{args.algo}_connect4_seed{seed}",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        print(f"zapis co {checkpoint_callback.save_freq} kroków")

        # --- TRENING ---
        model.learn(
            total_timesteps=16384,
            callback=checkpoint_callback,
            progress_bar=True,               # opcjonalnie
        )

        vec_env.close()

if __name__ == "__main__":
    main()
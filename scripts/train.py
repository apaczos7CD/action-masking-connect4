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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    # parser.add_argument(
    #     "--steps",
    #     type=int,
    #     default=200_000,
    #     help="Total number of training timesteps.",
    # )
    # parser.add_argument(
    #     "--vec-env-n",
    #     type=int,
    #     default=16,
    #     help="Number of parallel environments.",
    # )
    # parser.add_argument(
    #     "--n-steps",
    #     type=int,
    #     default=1_024,
    #     help="N steps.",
    # )
    # parser.add_argument(
    #     "--batch-size",
    #     type=int,
    #     default=1_024,
    #     help="batch size.",
    # )
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

    env_fns = [make_env(args.seed + i, args.algo) for i in range(16)]
    vec_env = DummyVecEnv(env_fns)

    model = create_model(algo=args.algo, vec_env=vec_env, seed=args.seed)

    checkpoint_callback = CheckpointCallback(
        save_freq=1024,
        save_path="checkpoints",
        name_prefix=f"{args.algo}_connect4_seed{args.seed}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print(f"zapis co {checkpoint_callback.save_freq} kroków")

    # --- TRENING ---
    model.learn(
        total_timesteps=32768,
        callback=checkpoint_callback,
        progress_bar=True,               # opcjonalnie
    )

    # --- ZAPIS KOŃCOWY ---
    # model.save(f"checkpoints\\{args.algo}_connect4_final_seed{args.seed}")

    vec_env.close()

if __name__ == "__main__":
    main()
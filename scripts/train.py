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
    parser.add_argument(
        "--steps",
        type=int,
        default=200_000,
        help="Total number of training timesteps.",
    )
    parser.add_argument(
        "--vec-env-n",
        type=int,
        default=16,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1_024,
        help="N steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_024,
        help="batch size.",
    )
    return parser.parse_args()


args = parse_args()

def mask_fn(env) -> np.ndarray:
    return env.action_masks()


def make_env(eval_seed: int, algo: str):
    def _factory():
        env = OneAgentVsRandomGym(seed=eval_seed)
        if algo == "maskable_ppo":
            env = ActionMasker(env, mask_fn)
        return env
    return _factory

# --- VEC ENV ---
env_fns = [make_env(args.seed + i) for i in range(args.vec_env_n)]
vec_env = DummyVecEnv(env_fns)

# --- MODEL ---
ModelCls = MaskablePPO if args.algo == "maskable_ppo" else PPO

model = ModelCls(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[64, 64]),
    tensorboard_log="runs",
    seed=args.seed,
    verbose=1,
)

checkpoint_callback = CheckpointCallback(
    save_freq=args.n_steps,
    save_path="checkpoints",
    name_prefix=f"{args.algo}_connect4_seed{args.seed}",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

print(f"zapis co {checkpoint_callback.save_freq} kroków")

# --- TRENING ---
model.learn(
    total_timesteps=args.steps,
    callback=checkpoint_callback,
    progress_bar=True,               # opcjonalnie
)

# --- ZAPIS KOŃCOWY ---
model.save(f"checkpoints\\{args.algo}_connect4_final_seed{args.seed}")

vec_env.close()
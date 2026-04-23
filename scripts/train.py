from pathlib import Path
import yaml
import numpy as np
import argparse
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.common.wrappers import ActionMasker

from envs.connect4_wrapper import OneAgentVsRandomGym


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_ppo.yaml"),
        help="Ścieżka do pliku YAML z konfiguracją",
    )
    return parser.parse_args()


def read_config(config):
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    return config


def mask_fn(env) -> np.ndarray:
    return env.action_masks()


def make_env(seed: int, algo: str):
    def _factory():
        env = OneAgentVsRandomGym(seed=seed)
        if algo == "maskable_ppo":
            env = ActionMasker(env, mask_fn)
        return env
    return _factory


def create_model(algo: str, seed:int, vec_env: DummyVecEnv, model_param):
    # --- MODEL ---
    modelCls = MaskablePPO if algo == "maskable_ppo" else PPO

    model = modelCls(
        model_param["policy"],
        vec_env,
        learning_rate=model_param["learning_rate"],
        n_steps=model_param["n_steps"],
        batch_size=model_param["batch_size"],
        gamma=model_param["gamma"],
        gae_lambda=model_param["gae_lambda"],
        clip_range=model_param["clip_range"],
        ent_coef=model_param["ent_coef"],
        vf_coef=model_param["vf_coef"],
        max_grad_norm=model_param["max_grad_norm"],
        policy_kwargs=dict(net_arch=model_param["net_arch"]),
        tensorboard_log=model_param["logs_dir"],
        seed=seed,
        verbose=1,
    )

    return model


def train_model(config):
    for seed in config["seeds"]:
        env_fns = [make_env(seed + i*10, config["algo"]) for i in range(config["vec_env_n"])]
        vec_env = DummyVecEnv(env_fns)

        model = create_model(algo=config["algo"], vec_env=vec_env, seed=seed, model_param=config["model_param"])

        checkpoint_callback = CheckpointCallback(
            save_freq=config["checkpoints_freq"],
            save_path=config["checkpoints_dir"],
            name_prefix=f"{config["algo"]}_connect4_seed{seed}",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        print(f"zapis co {checkpoint_callback.save_freq} kroków")

        # --- TRENING ---
        model.learn(
            total_timesteps=config["total_env_steps"],
            callback=checkpoint_callback,
            progress_bar=True,
        )

        vec_env.close()


def main():
    args = parse_args()
    config = read_config(args.config)

    #trening
    train_model(config)


if __name__ == "__main__":
    main()
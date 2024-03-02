import random
import os
import numpy as np
import torch

from evaluate import evaluate_HIV, evaluate_HIV_population
from train import ProjectAgent  # Replace DummyAgent with your agent implementation


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(seed=42)
    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    config = {'nb_actions': 4,
            'learning_rate': 0.001,
            'gamma':0.99,
            'buffer_size': 1000000,
            'epsilon_min': 0.1,
            'epsilon_max': 1.,
            'epsilon_decay_period': 40000,
            'epsilon_delay_decay': 1200,
            'batch_size': 1024,
            'gradient_steps': 5,
            'update_target_strategy': 'ema',#'replace', # or 'ema'
            'update_target_freq': 200,
            'update_target_tau': 0.001,
            'criterion': torch.nn.SmoothL1Loss(),
            'nb_neurons': 256,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'memory_length':5}

    agent = ProjectAgent(config)
    agent.load('qnetwork.pt')
    # Keep the following lines to evaluate your agent unchanged.
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
    with open(file="score.txt", mode="w") as f:
        f.write(f"{score_agent}\n{score_agent_dr}")

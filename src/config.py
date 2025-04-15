"""Configuration for the Chess RL project."""

import os
from datetime import datetime

# Default paths - using proper absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Go up one level to get the project root

# Three distinct model directories
DEFAULT_EXPERT_MODEL_DIR = os.path.join(PROJECT_ROOT, "expert_models")
DEFAULT_PRETRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT, "pretrain_models")
DEFAULT_TRAINED_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "trained_models", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

# Path to stockfish executable
DEFAULT_STOCKFISH_PATH = os.path.join(
    DEFAULT_EXPERT_MODEL_DIR, "stockfish/stockfish-ubuntu-x86-64-avx2"
)

# Training configuration
TRAINING_CONFIG = {
    "episodes": 1000,
    "max_steps": 100,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "learning_rate": 0.001,
    "stockfish_path": DEFAULT_STOCKFISH_PATH,
    "expert_model_dir": DEFAULT_EXPERT_MODEL_DIR,
    "pretrained_model_dir": DEFAULT_PRETRAINED_MODEL_DIR,
    "trained_output_dir": DEFAULT_TRAINED_OUTPUT_DIR,
    "save_freq": 100,
    "load_pretrained": None,
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "load_model": None,
    "eval_games": 10,
    "max_steps": 200,
    "render": False,
    "compare_expert": False,
    "stockfish_path": DEFAULT_STOCKFISH_PATH,
}


def get_args_parser():
    """Create and return argument parser for command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Chess RL Agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train the agent")
    train_parser.add_argument(
        "--episodes",
        type=int,
        default=TRAINING_CONFIG["episodes"],
        help="Number of episodes to train",
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=TRAINING_CONFIG["max_steps"],
        help="Maximum steps per episode",
    )
    train_parser.add_argument(
        "--gamma", type=float, default=TRAINING_CONFIG["gamma"], help="Discount factor"
    )
    train_parser.add_argument(
        "--epsilon",
        type=float,
        default=TRAINING_CONFIG["epsilon"],
        help="Initial exploration rate",
    )
    train_parser.add_argument(
        "--epsilon-min",
        type=float,
        default=TRAINING_CONFIG["epsilon_min"],
        help="Minimum exploration rate",
    )
    train_parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=TRAINING_CONFIG["epsilon_decay"],
        help="Epsilon decay rate",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=TRAINING_CONFIG["learning_rate"],
        help="Learning rate",
    )
    train_parser.add_argument(
        "--stockfish-path",
        type=str,
        default=TRAINING_CONFIG["stockfish_path"],
        help="Path to Stockfish engine",
    )
    train_parser.add_argument(
        "--expert-model-dir",
        type=str,
        default=TRAINING_CONFIG["expert_model_dir"],
        help="Directory with expert models (like Stockfish)",
    )
    train_parser.add_argument(
        "--pretrained-model-dir",
        type=str,
        default=TRAINING_CONFIG["pretrained_model_dir"],
        help="Directory with pretrained models to load",
    )
    train_parser.add_argument(
        "--trained-output-dir",
        type=str,
        default=TRAINING_CONFIG["trained_output_dir"],
        help="Directory to save trained models output",
    )
    train_parser.add_argument(
        "--save-freq",
        type=int,
        default=TRAINING_CONFIG["save_freq"],
        help="Save model every N episodes",
    )
    train_parser.add_argument(
        "--load-pretrained",
        type=str,
        default=TRAINING_CONFIG["load_pretrained"],
        help="Path to load existing pretrained model",
    )

    # Evaluation arguments
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the agent")
    eval_parser.add_argument(
        "--load-model", type=str, required=True, help="Path to trained model"
    )
    eval_parser.add_argument(
        "--eval-games",
        type=int,
        default=EVALUATION_CONFIG["eval_games"],
        help="Number of games to evaluate",
    )
    eval_parser.add_argument(
        "--max-steps",
        type=int,
        default=EVALUATION_CONFIG["max_steps"],
        help="Maximum steps per game",
    )
    eval_parser.add_argument(
        "--render", action="store_true", help="Render the board during evaluation"
    )
    eval_parser.add_argument(
        "--compare-expert", action="store_true", help="Compare with expert"
    )
    eval_parser.add_argument(
        "--stockfish-path",
        type=str,
        default=EVALUATION_CONFIG["stockfish_path"],
        help="Path to Stockfish engine",
    )

    return parser

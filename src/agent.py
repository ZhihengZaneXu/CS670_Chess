import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical

from .environment import ChessEnv
from .network import ChessNetwork
from .utils import board_to_tensor, move_to_direction_idx, selections_to_move


class ChessFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for the chess board state"""

    def __init__(self, observation_space, features_dim=256):
        super(ChessFeatureExtractor, self).__init__(observation_space, features_dim)

        # Shared feature extractor (using the same architecture as your original model)
        self.shared_layers = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Process the board tensor
        return self.shared_layers(observations)


class ChessPolicy(ActorCriticPolicy):
    """Custom policy for hierarchical chess actions"""

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(ChessPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
            features_extractor_class=ChessFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )

        # Piece selection head (actor)
        self.piece_action_net = nn.Linear(self.features_dim, 64)

        # Move direction head (actor)
        self.move_action_net = nn.Linear(self.features_dim, 64)

        # Value network
        self.value_net = nn.Linear(self.features_dim, 1)

    def forward(self, obs, deterministic=False):
        """Forward pass in network"""
        features = self.extract_features(obs)

        # Get piece logits and move logits
        piece_logits = self.piece_action_net(features)
        move_logits = self.move_action_net(features)

        # Get value estimate
        values = self.value_net(features)

        # We'll treat the first 64 values of the action space as piece selection
        # and the next 64 as move selection
        piece_distribution = Categorical(logits=piece_logits)
        move_distribution = Categorical(logits=move_logits)

        if deterministic:
            piece_actions = torch.argmax(piece_logits, dim=1)
            move_actions = torch.argmax(move_logits, dim=1)
        else:
            piece_actions = piece_distribution.sample()
            move_actions = move_distribution.sample()

        # Combine into a single action
        # We're using a 128-dimensional action space (64 for piece + 64 for move)
        actions = torch.cat(
            [piece_actions.unsqueeze(1), move_actions.unsqueeze(1)], dim=1
        )

        log_probs = torch.cat(
            [
                piece_distribution.log_prob(piece_actions).unsqueeze(1),
                move_distribution.log_prob(move_actions).unsqueeze(1),
            ],
            dim=1,
        )

        return actions, values, log_probs


class ChessRLAgent:
    def __init__(
        self,
        expert_model,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
        learning_rate=0.0003,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        clip_range=0.2,
        device="auto",
    ):
        """
        Initialize the chess RL agent with PPO

        Args:
            expert_model: Expert model (mentor) to imitate
            gamma: Discount factor
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            learning_rate: Learning rate
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum norm for gradient clipping
            gae_lambda: Factor for trade-off of bias vs variance for GAE
            clip_range: Clipping parameter for PPO
            device: Device to run the model on
        """
        self.expert = expert_model

        # Create a custom gym environment wrapping our ChessEnv
        self.env = ChessPPOEnv(self.expert)

        # Create the PPO model with our custom policy
        self.model = PPO(
            ChessPolicy,
            self.env,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            device=device,
            verbose=1,
        )

    def get_action(self, board):
        """
        Get action using the learned policy

        Args:
            board: chess.Board representation of the current state

        Returns:
            chess.Move: Selected action
        """
        # Convert board to tensor
        obs = board_to_tensor(board).numpy()

        # Get raw action from model
        raw_action, _ = self.model.predict(obs, deterministic=False)

        # Convert to chess move
        piece_selection = raw_action[0]
        move_selection = raw_action[1]

        valid_moves = list(board.legal_moves)

        if not valid_moves:
            return None

        # Convert piece and move selections to a valid chess move
        selected_move = selections_to_move(piece_selection, move_selection, valid_moves)

        # If we couldn't find a valid move, choose randomly
        if selected_move is None and valid_moves:
            selected_move = random.choice(valid_moves)

        return selected_move

    def train(self, total_timesteps=10000):
        """Train the PPO model"""
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, filepath):
        """Save the model to a file"""
        self.model.save(filepath)

    def load(self, filepath):
        """Load the model from a file"""
        self.model = PPO.load(filepath, env=self.env)


class ChessPPOEnv(gym.Env):
    """
    Custom gym environment wrapper for chess that works with Stable Baselines3
    """

    def __init__(self, expert_model):
        super(ChessPPOEnv, self).__init__()

        # Initialize the chess environment
        self.chess_env = ChessEnv()
        self.expert_model = expert_model

        # Define action and observation spaces
        # Action space: 64 squares for piece selection + 64 directions for move selection
        self.action_space = spaces.MultiDiscrete([64, 64])

        # Observation space: 12 channels (6 piece types × 2 colors) × 8×8 board
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(12, 8, 8), dtype=np.float32
        )

        self.current_board = None

    def reset(self):
        """Reset the environment"""
        self.current_board = self.chess_env.reset()
        return board_to_tensor(self.current_board).numpy()

    def step(self, action):
        """
        Execute an action in the environment

        Args:
            action: Tuple of (piece_selection, move_selection)

        Returns:
            tuple: (observation, reward, done, info)
        """
        piece_selection, move_selection = action

        # Convert to a chess move
        valid_moves = list(self.current_board.legal_moves)
        chess_move = selections_to_move(piece_selection, move_selection, valid_moves)

        # If we couldn't find a valid move, choose randomly
        if chess_move is None and valid_moves:
            chess_move = random.choice(valid_moves)

        # Get expert's move for reward calculation
        expert_move = None
        if self.expert_model:
            expert_move = self.expert_model.get_best_move(self.current_board)

        # Execute move in the chess environment
        next_board, env_reward, done, info = self.chess_env.step(chess_move)
        self.current_board = next_board

        # Calculate reward based on expert imitation
        reward = self._calculate_reward(self.current_board, chess_move, expert_move)

        # Return step information
        return board_to_tensor(self.current_board).numpy(), reward, done, info

    def _calculate_reward(self, board, action, expert_action):
        """
        Calculate reward based on similarity to expert action

        Args:
            board: Current board state
            action: Agent's selected action
            expert_action: Expert's selected action

        Returns:
            float: Reward value
        """
        if action is None or expert_action is None:
            return 0.0

        if action == expert_action:
            return 1.0  # Perfect match

        # Partial reward for similar moves
        from_square_match = action.from_square == expert_action.from_square
        to_square_match = action.to_square == expert_action.to_square

        if from_square_match:
            return 0.5  # Selected the same piece
        elif to_square_match:
            return 0.3  # Moved to the same square
        else:
            # Check if the piece type is the same
            piece_at_action = board.piece_at(action.from_square)
            piece_at_expert = board.piece_at(expert_action.from_square)

            if (
                piece_at_action
                and piece_at_expert
                and piece_at_action.piece_type == piece_at_expert.piece_type
            ):
                return 0.1  # At least moved the same type of piece

        return -0.1  # Different move

    def render(self, mode="human"):
        """Render the current board state"""
        return self.chess_env.render()

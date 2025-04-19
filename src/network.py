import random

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical


# Feature extractor for chess board state
class ChessFeatureExtractor(BaseFeaturesExtractor):
    """CNN feature extractor for the chess board state"""

    def __init__(self, observation_space, features_dim=256):
        super(ChessFeatureExtractor, self).__init__(observation_space, features_dim)

        # CNN for processing 12×8×8 board representation
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Handle the case when observations is a dictionary
        if isinstance(observations, dict):
            # Extract just the board tensor from the dictionary
            board_tensor = observations["board"]
            return self.cnn(board_tensor)
        else:
            # For non-dictionary observations, process directly
            return self.cnn(observations)


# Hierarchical policy for chess (piece selection then move selection)
class HierarchicalChessPolicy(ActorCriticPolicy):
    """Custom policy for hierarchical chess actions"""

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # Set up feature extractor
        kwargs.update(
            {
                "features_extractor_class": ChessFeatureExtractor,
                "features_extractor_kwargs": dict(features_dim=256),
            }
        )

        super(HierarchicalChessPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        # First action head: piece selection (64 squares on board)
        self.piece_action_net = nn.Linear(self.features_dim, 64)

        # Embedding layer for the selected piece
        self.piece_embedding = nn.Embedding(64, 64)

        # Second action head: move direction (takes features + piece embedding)
        self.move_action_net = nn.Linear(self.features_dim + 64, 64)

        # Value network
        self.value_net = nn.Linear(self.features_dim, 1)

    def forward(self, obs, deterministic=False):
        """Forward pass in network"""
        features = self.extract_features(obs)

        # First action: piece selection
        piece_logits = self.piece_action_net(features)

        # If piece masks are provided in the observation
        if isinstance(obs, dict) and "piece_mask" in obs:
            # Apply mask to piece logits (set invalid to -inf)
            piece_mask = obs["piece_mask"]
            masked_piece_logits = piece_logits.clone()
            masked_piece_logits[piece_mask == 0] = float("-inf")
            piece_distribution = Categorical(logits=masked_piece_logits)
        else:
            # No masks available, use raw logits
            piece_distribution = Categorical(logits=piece_logits)

        # Sample piece action
        if deterministic:
            piece_actions = torch.argmax(piece_logits, dim=1)
        else:
            piece_actions = piece_distribution.sample()

        # Get embedding for the selected piece
        piece_embed = self.piece_embedding(piece_actions)

        # Combine features with piece embedding for second action
        combined_features = torch.cat([features, piece_embed], dim=1)

        # Second action: move selection based on the selected piece
        move_logits = self.move_action_net(combined_features)

        # If move masks are provided in the observation
        if isinstance(obs, dict) and "move_mask" in obs:
            # Apply dynamic move mask based on selected piece
            # This requires gathering the appropriate masks for each selected piece
            batch_size = piece_actions.shape[0]
            move_masks = torch.zeros_like(move_logits)

            for i in range(batch_size):
                piece_idx = piece_actions[i].item()
                # Get the move mask for this piece
                # This assumes move_mask is a tensor of shape (batch_size, 64, 64)
                # where the second dimension is the piece index
                if piece_idx < obs["move_mask"].shape[1]:
                    move_masks[i] = obs["move_mask"][i, piece_idx]

            # Apply mask
            masked_move_logits = move_logits.clone()
            masked_move_logits[move_masks == 0] = float("-inf")
            move_distribution = Categorical(logits=masked_move_logits)
        else:
            # No masks available
            move_distribution = Categorical(logits=move_logits)

        # Sample move action
        if deterministic:
            move_actions = torch.argmax(move_logits, dim=1)
        else:
            move_actions = move_distribution.sample()

        # Combine actions
        actions = torch.stack([piece_actions, move_actions], dim=1)

        # Calculate log probabilities
        log_prob_piece = piece_distribution.log_prob(piece_actions)
        log_prob_move = move_distribution.log_prob(move_actions)
        combined_log_prob = log_prob_piece + log_prob_move

        # Value estimate directly from features
        values = self.value_net(features)

        return actions, values, combined_log_prob.unsqueeze(1)

    def evaluate_actions(self, obs, actions):
        """Evaluate actions for PPO loss calculation"""
        features = self.extract_features(obs)

        # Split actions into piece and move
        piece_actions = actions[:, 0].long()
        move_actions = actions[:, 1].long()

        # Get piece logits
        piece_logits = self.piece_action_net(features)

        # Apply masks if available
        if isinstance(obs, dict) and "piece_mask" in obs:
            masked_piece_logits = piece_logits.clone()
            masked_piece_logits[obs["piece_mask"] == 0] = float("-inf")
            piece_distribution = Categorical(logits=masked_piece_logits)
        else:
            piece_distribution = Categorical(logits=piece_logits)

        # Get piece embedding
        piece_embed = self.piece_embedding(piece_actions)

        # Combine features with piece embedding
        combined_features = torch.cat([features, piece_embed], dim=1)

        # Get move logits
        move_logits = self.move_action_net(combined_features)

        # Apply move masks if available
        if isinstance(obs, dict) and "move_mask" in obs:
            batch_size = piece_actions.shape[0]
            move_masks = torch.zeros_like(move_logits)

            for i in range(batch_size):
                piece_idx = piece_actions[i].item()
                if piece_idx < obs["move_mask"].shape[1]:
                    move_masks[i] = obs["move_mask"][i, piece_idx]

            masked_move_logits = move_logits.clone()
            masked_move_logits[move_masks == 0] = float("-inf")
            move_distribution = Categorical(logits=masked_move_logits)
        else:
            move_distribution = Categorical(logits=move_logits)

        # Calculate log probabilities
        log_prob_piece = piece_distribution.log_prob(piece_actions)
        log_prob_move = move_distribution.log_prob(move_actions)
        combined_log_prob = log_prob_piece + log_prob_move

        # Calculate entropy (for exploration)
        entropy = (
            piece_distribution.entropy().mean() + move_distribution.entropy().mean()
        )

        # Get value estimate
        values = self.value_net(features)

        return values, combined_log_prob, entropy

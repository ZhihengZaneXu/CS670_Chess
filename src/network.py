import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical


class ChessFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(ChessFeatureExtractor, self).__init__(observation_space, features_dim)
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
        board = (
            observations["board"] if isinstance(observations, dict) else observations
        )
        return self.cnn(board)


class HierarchicalChessPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        kwargs.update(
            {
                "features_extractor_class": ChessFeatureExtractor,
                "features_extractor_kwargs": dict(features_dim=64),
            }
        )
        super(HierarchicalChessPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        self.piece_action_net = nn.Linear(self.features_dim, 64)
        self.piece_embedding = nn.Embedding(64, 64)
        self.move_action_net = nn.Linear(self.features_dim + 64, 64)

        self.value_net = nn.Linear(self.features_dim, 1)

    def forward(self, obs, deterministic=False):
        # Feature extraction
        features = self.extract_features(obs)

        # === Piece selection head ===
        piece_logits = self.piece_action_net(features)
        if isinstance(obs, dict) and "piece_mask" in obs:
            mask = obs["piece_mask"].float()
            piece_logits = piece_logits.masked_fill(mask == 0, float("-1e9"))
        piece_probs = F.softmax(piece_logits, dim=-1)
        piece_dist = Categorical(probs=piece_probs)

        if deterministic:
            piece_actions = torch.argmax(piece_logits, dim=1)
        else:
            piece_actions = piece_dist.sample()

        # === Move selection head ===
        piece_embed = self.piece_embedding(piece_actions)
        combined_features = torch.cat([features, piece_embed], dim=1)
        move_logits = self.move_action_net(combined_features)
        if isinstance(obs, dict) and "move_mask" in obs:
            batch_size = move_logits.size(0)
            move_masks = torch.zeros_like(move_logits)
            for i in range(batch_size):
                pi = piece_actions[i].item()
                move_masks[i] = obs["move_mask"][i, pi]
            move_logits = move_logits.masked_fill(move_masks == 0, float("-1e9"))
        move_probs = F.softmax(move_logits, dim=-1)
        move_dist = Categorical(probs=move_probs)
        if deterministic:
            move_actions = torch.argmax(move_logits, dim=1)
        else:
            move_actions = move_dist.sample()

        # === Pack outputs ===
        actions = torch.stack([piece_actions, move_actions], dim=1)
        log_prob_piece = piece_dist.log_prob(piece_actions)
        log_prob_move = move_dist.log_prob(move_actions)
        combined_log_prob = (log_prob_piece + log_prob_move).unsqueeze(1)
        values = self.value_net(features)
        return actions, values, combined_log_prob

    def evaluate_actions(self, obs, actions):
        # Similar masking + softmax logic for evaluation
        features = self.extract_features(obs)

        # Unpack actions
        piece_actions = actions[:, 0].long()
        move_actions = actions[:, 1].long()

        # Piece logits + mask
        piece_logits = self.piece_action_net(features)
        if isinstance(obs, dict) and "piece_mask" in obs:
            mask = obs["piece_mask"].float()
            piece_logits = piece_logits.masked_fill(mask == 0, float("-1e9"))
        piece_probs = F.softmax(piece_logits, dim=-1)
        piece_dist = Categorical(probs=piece_probs)
        log_prob_piece = piece_dist.log_prob(piece_actions)

        # Move logits + mask
        piece_embed = self.piece_embedding(piece_actions)
        combined = torch.cat([features, piece_embed], dim=1)
        move_logits = self.move_action_net(combined)
        if isinstance(obs, dict) and "move_mask" in obs:
            batch_size = move_logits.size(0)
            move_masks = torch.zeros_like(move_logits)
            for i in range(batch_size):
                pi = piece_actions[i].item()
                move_masks[i] = obs["move_mask"][i, pi]
            move_logits = move_logits.masked_fill(move_masks == 0, float("-1e9"))
        move_probs = F.softmax(move_logits, dim=-1)
        move_dist = Categorical(probs=move_probs)
        log_prob_move = move_dist.log_prob(move_actions)

        # Value & entropy
        values = self.value_net(features)
        entropy = piece_dist.entropy().mean() + move_dist.entropy().mean()
        return values, (log_prob_piece + log_prob_move).unsqueeze(1), entropy

import os
import random

import chess
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class FrequentLoggingCallback(BaseCallback):
    """
    Logs a counter every `log_freq` steps so you get more frequent console/TB feedback.
    """

    def __init__(self, log_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.log_freq == 0:
            # for example, log the current global step count
            self.logger.record("train/step", self.step_count)
        return True


def board_to_tensor(board):
    """Convert a chess.Board to a 12×8×8 tensor representation."""
    tensor = torch.zeros(12, 8, 8)
    piece_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = divmod(square, 8)
            channel = piece_to_channel[piece.piece_type] + (0 if piece.color else 6)
            tensor[channel, 7 - rank, file] = 1.0
    return tensor


def selections_to_move(piece_selection, move_selection, valid_moves):
    """Convert piece and move indices to a legal chess.Move, or None."""
    moves_by_source = {}
    for move in valid_moves:
        moves_by_source.setdefault(move.from_square, []).append(move)
    possible = moves_by_source.get(piece_selection)
    if not possible:
        return None
    if len(possible) == 1:
        return possible[0]
    idx = min(move_selection, len(possible) - 1)
    return possible[idx]


def create_chess_agent(
    env,
    gamma=0.99,
    n_steps=512,
    batch_size=64,
    learning_rate=3e-4,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=None,
    device="auto",
):
    """Instantiate a PPO agent with the hierarchical chess policy."""
    log_dir = tensorboard_log or os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    from network import HierarchicalChessPolicy

    model = PPO(
        HierarchicalChessPolicy,
        env,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        device=device,
        verbose=1,
        tensorboard_log=log_dir,
    )
    return model

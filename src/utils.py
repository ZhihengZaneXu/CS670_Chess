import os
import random

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


def move_to_direction_idx(move):
    """
    Convert a chess.Move to a direction-distance index (0-55)
    Plus special cases for knight moves (56-63)
    """
    from_square = move.from_square
    to_square = move.to_square

    from_row, from_col = divmod(from_square, 8)
    to_row, to_col = divmod(to_square, 8)

    # Calculate delta
    row_delta = to_row - from_row
    col_delta = to_col - from_col

    # Special case for knight moves
    if (abs(row_delta) == 2 and abs(col_delta) == 1) or (
        abs(row_delta) == 1 and abs(col_delta) == 2
    ):
        # Knights have special encoding from 56-63
        if row_delta == 2 and col_delta == 1:
            return 56
        elif row_delta == 2 and col_delta == -1:
            return 57
        elif row_delta == -2 and col_delta == 1:
            return 58
        elif row_delta == -2 and col_delta == -1:
            return 59
        elif row_delta == 1 and col_delta == 2:
            return 60
        elif row_delta == 1 and col_delta == -2:
            return 61
        elif row_delta == -1 and col_delta == 2:
            return 62
        elif row_delta == -1 and col_delta == -2:
            return 63

    # For other pieces
    directions = [
        (1, 0),  # N
        (1, 1),  # NE
        (0, 1),  # E
        (-1, 1),  # SE
        (-1, 0),  # S
        (-1, -1),  # SW
        (0, -1),  # W
        (1, -1),  # NW
    ]

    for dir_idx, (dr, dc) in enumerate(directions):
        # Check if move is in this direction
        if row_delta == 0 or col_delta == 0 or abs(row_delta) == abs(col_delta):
            # Calculate direction
            if row_delta != 0:
                r_sign = row_delta // abs(row_delta)
            else:
                r_sign = 0

            if col_delta != 0:
                c_sign = col_delta // abs(col_delta)
            else:
                c_sign = 0

            if (r_sign, c_sign) == (dr, dc):
                # It's in this direction, now calculate the distance
                distance = max(abs(row_delta), abs(col_delta))
                if 1 <= distance <= 7:
                    return dir_idx * 7 + (distance - 1)

    # If we can't encode this move in our direction-distance model
    # (like castling or pawn promotion), return None
    return None


# Utility functions for converting between chess moves and network actions
def board_to_tensor(board):
    """Convert a chess.Board to a 12×8×8 tensor representation"""
    import chess
    import torch

    # Initialize tensor
    tensor = torch.zeros(12, 8, 8)

    # Map each piece to the corresponding channel
    piece_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # Fill tensor with pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank, file = divmod(square, 8)
            channel = piece_to_channel[piece.piece_type]
            if not piece.color:  # Black pieces go in channels 6-11
                channel += 6
            tensor[channel, 7 - rank, file] = 1.0

    return tensor


def selections_to_move(piece_selection, move_selection, valid_moves):
    """Convert piece and move selections to a chess.Move"""
    import chess

    # Group valid moves by source square
    moves_by_source = {}
    for move in valid_moves:
        source = move.from_square
        if source not in moves_by_source:
            moves_by_source[source] = []
        moves_by_source[source].append(move)

    # If piece selection is valid and has legal moves
    if piece_selection in moves_by_source:
        possible_moves = moves_by_source[piece_selection]

        # If we have exactly one move, return it
        if len(possible_moves) == 1:
            return possible_moves[0]

        # Try to match the move_selection to a direction
        # Simplified version: just use the index if in range
        if move_selection < len(possible_moves):
            return possible_moves[move_selection]
        else:
            # Fallback to a random valid move for this piece
            return random.choice(possible_moves)

    # Fallback if piece selection is not valid
    return None


def create_chess_agent(
    env,
    expert_model=None,
    gamma=0.99,
    n_steps=512,
    batch_size=64,
    learning_rate=0.0003,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./chess_tb_logs/",
    device="auto",
):
    """Create and return a PPO agent for chess"""
    os.makedirs(tensorboard_log, exist_ok=True)

    # Import within function to avoid circular imports
    from network import HierarchicalChessPolicy

    # Create the PPO model with our hierarchical policy
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
        tensorboard_log=tensorboard_log,
    )

    return model


def train_chess_agent(
    model, total_timesteps=10000, save_path="./chess_model_checkpoints/"
):
    """Train the PPO model with logging and checkpointing"""
    os.makedirs(save_path, exist_ok=True)

    # Import within function to avoid circular imports
    from network import FrequentLoggingCallback

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=save_path, name_prefix="chess_model"
    )

    # For more frequent logging
    frequent_logging = FrequentLoggingCallback(log_freq=10)

    # Combine callbacks
    callbacks = [checkpoint_callback, frequent_logging]

    # Train with callbacks
    model.learn(
        total_timesteps=total_timesteps, callback=callbacks, tb_log_name="chess_run"
    )

    return model


# Callback for frequent logging to TensorBoard
class FrequentLoggingCallback(BaseCallback):
    """Callback for logging metrics more frequently"""

    def __init__(self, verbose=0, log_freq=100):
        super(FrequentLoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            self.logger.dump(self.num_timesteps)
        return True

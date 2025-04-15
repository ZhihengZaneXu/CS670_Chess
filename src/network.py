import random

import torch
import torch.nn as nn

from .utils import board_to_tensor, move_to_direction_idx, selections_to_move


class ChessNetwork(nn.Module):
    def __init__(self, hidden_size=256):
        super(ChessNetwork, self).__init__()
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden_size),
            nn.ReLU(),
        )

        # Piece selection head
        self.piece_head = nn.Linear(hidden_size, 64)  # 64 squares on the board

        # Move direction head
        # 8 directions × 7 distances + 8 knight moves = 64 possible moves
        self.move_head = nn.Linear(hidden_size, 64)

    def forward(self, x):
        features = self.shared_layers(x)
        piece_logits = self.piece_head(features)
        move_logits = self.move_head(features)
        return piece_logits, move_logits

    def get_action(self, board, valid_moves):
        """
        Select an action using the network and apply a mask for valid moves
        """
        state_tensor = board_to_tensor(board).unsqueeze(0)  # Add batch dimension
        piece_logits, move_logits = self(state_tensor)

        # Create a mask for valid piece selections
        piece_mask = torch.zeros(64)

        # Group moves by source square
        valid_sources = set()
        for move in valid_moves:
            valid_sources.add(move.from_square)

        for square in valid_sources:
            piece_mask[square] = 1.0

        # Apply mask to piece logits (set invalid pieces to -inf)
        masked_piece_logits = piece_logits.clone()
        masked_piece_logits[0, piece_mask == 0] = float("-inf")

        # Get piece selection using softmax
        piece_probs = nn.functional.softmax(masked_piece_logits, dim=1)
        piece_selection = torch.multinomial(piece_probs[0], 1).item()

        # Now for the selected piece, find valid moves
        valid_directions = []
        for move in valid_moves:
            if move.from_square == piece_selection:
                # Convert move to direction-distance encoding
                direction_idx = move_to_direction_idx(move)
                if direction_idx is not None:
                    valid_directions.append(direction_idx)

        # Create a mask for valid directions
        direction_mask = torch.zeros(64)
        for idx in valid_directions:
            direction_mask[idx] = 1.0

        # Apply mask to move logits
        masked_move_logits = move_logits.clone()
        masked_move_logits[0, direction_mask == 0] = float("-inf")

        # Get direction selection using softmax
        direction_probs = nn.functional.softmax(masked_move_logits, dim=1)

        # Handle the case where there are no valid directions
        if len(valid_directions) == 0:
            # Choose a different piece or a random move
            return random.choice(list(valid_moves)) if valid_moves else None

        direction_selection = torch.multinomial(direction_probs[0], 1).item()

        # Convert piece and direction selections back to a chess move
        selected_move = selections_to_move(
            piece_selection, direction_selection, valid_moves
        )

        return selected_move

import chess
import torch


def board_to_tensor(board):
    """
    Convert a chess.Board to a tensor representation.

    Returns:
        torch.Tensor: Shape (12, 8, 8) representing the board state
    """
    pieces = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]
    colors = [chess.WHITE, chess.BLACK]

    tensor = torch.zeros(12, 8, 8)

    for color_idx, color in enumerate(colors):
        for piece_idx, piece in enumerate(pieces):
            channel_idx = piece_idx + 6 * color_idx
            for square in chess.SquareSet(board.pieces(piece, color)):
                row, col = divmod(square, 8)
                tensor[channel_idx, row, col] = 1.0

    return tensor


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


def selections_to_move(piece_selection, direction_selection, valid_moves):
    """
    Convert piece selection and direction selection to a chess.Move
    """
    import random

    if direction_selection >= 56:  # Knight move
        from_square = piece_selection

        # Knight moves encoding
        knight_offsets = [
            (2, 1),
            (2, -1),
            (-2, 1),
            (-2, -1),
            (1, 2),
            (1, -2),
            (-1, 2),
            (-1, -2),
        ]

        offset_idx = direction_selection - 56
        dr, dc = knight_offsets[offset_idx]

        from_row, from_col = divmod(from_square, 8)
        to_row = from_row + dr
        to_col = from_col + dc

        if 0 <= to_row < 8 and 0 <= to_col < 8:
            to_square = to_row * 8 + to_col

            # Find matching move in valid_moves
            for move in valid_moves:
                if move.from_square == from_square and move.to_square == to_square:
                    return move
    else:
        # Regular directional move
        dir_idx = direction_selection // 7
        distance = (direction_selection % 7) + 1

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

        dr, dc = directions[dir_idx]
        from_square = piece_selection
        from_row, from_col = divmod(from_square, 8)

        to_row = from_row + dr * distance
        to_col = from_col + dc * distance

        if 0 <= to_row < 8 and 0 <= to_col < 8:
            to_square = to_row * 8 + to_col

            # Find matching move in valid_moves
            for move in valid_moves:
                if move.from_square == from_square and move.to_square == to_square:
                    return move

    # If no move matches, return a random valid move
    return random.choice(list(valid_moves)) if valid_moves else None

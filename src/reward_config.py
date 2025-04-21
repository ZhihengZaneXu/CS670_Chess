# reward_config.py

import chess

# ── Terminal outcome weights ───────────────────────────────────────────────
WIN_REWARD = 10  # reward for agent checkmating opponent
DRAW_REWARD = 3  # reward for draw
LOSS_PENALTY = -3.0  # penalty for agent being checkmated

# ── Per‐move shaping ──────────────────────────────────────────────────────
SHORT_GAME_PENALTY = -0.01  # baseline per‐move penalty to encourage shorter games

# Imitation learning bonuses (if using an expert model)
IMIT_PERFECT_MOVE = 9  # agent exactly matches expert
IMIT_FROM_MATCH = 3  # correct piece selected
IMIT_TO_MATCH = 3  # correct destination square
IMIT_TYPE_MATCH = 0.5  # moved same type of piece
IMIT_MISMATCH_PEN = -0.1  # penalty for a completely different move

# ── Piece‐capture rewards ─────────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
CAPTURE_SCALE = 0.5  # multiply by PIECE_VALUES[piece_type]

# ── Post‐opponent material balance ─────────────────────────────────────────
MATERIAL_SCALE = 0.01  # reward ~= MATERIAL_SCALE * (our_material - their_material)

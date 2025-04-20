# reward_config.py

import chess

# ── Terminal outcome weights ───────────────────────────────────────────────
WIN_REWARD = 3  # reward for agent checkmating opponent
DRAW_REWARD = 1  # reward for draw
LOSS_PENALTY = -1.0  # penalty for agent being checkmated

# ── Per‐move shaping ──────────────────────────────────────────────────────
SHORT_GAME_PENALTY = -0.01  # baseline per‐move penalty to encourage shorter games

# Imitation learning bonuses (if using an expert model)
IMIT_PERFECT_MOVE = 3  # agent exactly matches expert
IMIT_FROM_MATCH = 1  # correct piece selected
IMIT_TO_MATCH = 1  # correct destination square
IMIT_TYPE_MATCH = 0.5  # moved same type of piece
IMIT_MISMATCH_PEN = -0.1  # penalty for a completely different move

# ── Piece‐capture rewards ─────────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 2,
    chess.QUEEN: 3,
}
CAPTURE_SCALE = 0.5  # multiply by PIECE_VALUES[piece_type]

# ── Post‐opponent material balance ─────────────────────────────────────────
MATERIAL_SCALE = 0.01  # reward ~= MATERIAL_SCALE * (our_material - their_material)

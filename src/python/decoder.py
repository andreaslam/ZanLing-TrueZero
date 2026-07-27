"""Python port of the Rust network input/output decoding (`src/decoder.rs`).

This module converts a python-chess board into the 21-plane canonical network
input tensor, and decodes raw network outputs (value, policy) into per-move
probabilities over the legal moves.

Coordinate / perspective contract (see ENGINE_SEMANTIC_SPEC.md):
- Piece planes: first 6 are the side to move ("us"), next 6 the opponent,
  each ordered pawn, knight, bishop, rook, queen, king.
- If Black is to move, every board square is rotated 180 degrees
  (flat index i -> 63 - i); White to move keeps physical squares.
- The en-passant plane uses the same square rotation.
- Castling scalar channels are ordered us (long, short) then opponent (long,
  short). Channel 0/1 are white-to-move / black-to-move indicators, channel 6
  is the repetition count, channel 7 the halfmove clock.
- Policy indices live in the fixed White-absolute 1880-move space defined by
  `src/mvs.rs`. Black moves are rank-flipped (mirror, not rotation) before
  lookup, exactly like the Rust `extract_policy`.
"""

import math
import os

import chess
import torch

# ---------------------------------------------------------------------------
# Move table (mirror `src/mvs.rs` CONTENTS_COZY exactly, via `list.txt`)
# ---------------------------------------------------------------------------


def _load_move_list():
    """Load the 1880-entry White-absolute move list from `list.txt`.

    `list.txt` holds the exact ordered move strings (UCI) matching
    `src/mvs.rs` CONTENTS_COZY, one per line. Promotions carry a trailing
    piece letter (q/r/b/n).
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "list.txt")
    with open(path, encoding="utf-8") as f:
        moves = [chess.Move.from_uci(line.strip()) for line in f if line.strip()]
    return moves


CONTENTS = _load_move_list()
assert len(CONTENTS) == 1880, f"expected 1880 moves, got {len(CONTENTS)}"
assert len(set(CONTENTS)) == 1880, "duplicate moves in table"

# move -> flat policy index (mirrors the Rust `contents.iter().position` lookup)
MOVE_TO_INDEX = {mv: i for i, mv in enumerate(CONTENTS)}

# piece plane order shared by both sides: pawn .. king (matches cozy-chess Piece::ALL)
PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def _flip_rank(square):
    """Mirror a square vertically (rank flip), used for Black policy lookup."""
    return chess.square(chess.square_file(square), 7 - chess.square_rank(square))


# ---------------------------------------------------------------------------
# Board -> tensor  (port of Rust `board_data` + `convert_board`)
# ---------------------------------------------------------------------------


def convert_board(board, bigl=None):
    """Convert a python-chess `Board` into the canonical 21x8x8 input tensor.

    Returns `(tensor, bigl)`. `bigl` is kept only for backward compatibility
    with the old API and is always an empty tensor.
    """
    us = board.turn
    opponent = not us

    # --- scalar channels -------------------------------------------------
    scalars = [0.0] * 8
    if us == chess.WHITE:
        scalars[0] = 1.0
    else:
        scalars[1] = 1.0

    # castling rights, us first then opponent (long then short)
    scalars[2] = 1.0 if board.has_queenside_castling_rights(us) else 0.0
    scalars[3] = 1.0 if board.has_kingside_castling_rights(us) else 0.0
    scalars[4] = 1.0 if board.has_queenside_castling_rights(opponent) else 0.0
    scalars[5] = 1.0 if board.has_kingside_castling_rights(opponent) else 0.0

    # repetition count of the current position + halfmove clock
    scalars[6] = float(_repetition_count(board))
    scalars[7] = float(board.halfmove_clock)

    # --- piece planes ----------------------------------------------------
    planes = torch.zeros(13, 8, 8)
    counter = 0
    for color in (us, opponent):
        for piece in PIECE_ORDER:
            for sq in board.pieces(piece, color):
                if us == chess.BLACK:
                    sq = 63 - sq  # 180-degree rotation for Black to move
                planes[counter, sq // 8, sq % 8] = 1.0
            counter += 1

    # --- en-passant plane ------------------------------------------------
    # Mirrors Rust: White to move marks the victim square on Rank::Fourth
    # (rank index 3); Black to move marks Rank::Fifth (rank index 4) and
    # applies the same 180-degree rotation as the piece planes.
    if board.ep_square is not None:
        if us == chess.WHITE:
            ep_sq = chess.square(chess.square_file(board.ep_square), 3)
        else:
            ep_sq = 63 - chess.square(chess.square_file(board.ep_square), 4)
        planes[12, ep_sq // 8, ep_sq % 8] = 1.0

    # scalars are broadcast across the board and placed before the bool planes
    scalar_planes = (
        torch.tensor(scalars, dtype=torch.float32).view(8, 1, 1).expand(8, 8, 8)
    )
    tensor = torch.cat([scalar_planes, planes], dim=0)  # (21, 8, 8)

    if bigl is None:
        bigl = torch.tensor([])
    return tensor, bigl


def _repetition_count(board):
    """Number of times the current position occurred before (Rust `get_reps`).

    Uses python-chess's Zobrist transposition key, matching the Rust
    `BoardStack::get_reps` which hashes prior positions on the move stack.
    Pops and re-pushes from a snapshot of the move stack so the board is left
    unmodified.
    """
    target = board._transposition_key()
    stack = list(board.move_stack)
    count = 0
    for _ in range(len(stack)):
        board.pop()
        if board._transposition_key() == target:
            count += 1
    for move in stack:
        board.push(move)
    return count


# ---------------------------------------------------------------------------
# Network output -> legal policy  (port of Rust `extract_policy` + decode)
# ---------------------------------------------------------------------------


def extract_policy(board):
    """Return `(legal_moves, idx_li)` for the position.

    `legal_moves` are python-chess moves canonicalised into the White-absolute
    policy space (rank-flipped for Black to move); `idx_li` are their flat
    policy indices in the same order as the board's legal-move generator.
    """
    legal_moves = []
    for mv in board.legal_moves:
        if board.turn == chess.BLACK:
            mv = chess.Move(
                _flip_rank(mv.from_square),
                _flip_rank(mv.to_square),
                promotion=mv.promotion,
            )
        legal_moves.append(mv)

    idx_li = [MOVE_TO_INDEX[mv] for mv in legal_moves]
    return legal_moves, idx_li


def decode_nn_output(value, policy, board):
    """Decode raw network outputs for a position.

    `value` is the network's 5-element value head output
    `(scalar logit, wdl logit x3, moves-left)` and `policy` the 1880 logits.
    Both may be torch tensors or anything convertible; batched inputs are
    squeezed.

    Returns `(value, wdl, moves_left, legal_moves, legal_lookup, policy_list,
    best_move)`:
      - `value`: scalar in [-1, 1] (tanh of the raw logit), from the side to
        move's perspective;
      - `wdl`: `[win, draw, loss]` probabilities for the side to move;
      - `moves_left`: clamped to be non-negative;
      - `legal_moves`: canonicalised legal moves (White-absolute space);
      - `legal_lookup`: `{uci_string: softmax_probability}` for legal moves;
      - `policy_list`: probabilities aligned with `legal_moves`;
      - `best_move`: UCI string of the highest-probability legal move
        (physical coordinates for the side to move).
    """
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.float32)
    if not torch.is_tensor(policy):
        policy = torch.tensor(policy, dtype=torch.float32)

    value = value.squeeze().float().cpu()
    policy = policy.squeeze().float().cpu()

    raw = value.tolist()
    scalar = math.tanh(raw[0])

    wdl_logits = torch.tensor(raw[1:4], dtype=torch.float32)
    wdl = torch.softmax(wdl_logits, dim=0).tolist()

    moves_left = max(0.0, raw[4])

    legal_moves, idx_li = extract_policy(board)
    legal_logits = policy[idx_li]
    policy_probs = torch.softmax(legal_logits, dim=0).tolist()

    legal_lookup = {mv.uci(): p for mv, p in zip(legal_moves, policy_probs)}

    # best move in physical coordinates for the side to move
    best_canonical = legal_moves[
        max(range(len(legal_moves)), key=lambda i: policy_probs[i])
    ]
    if board.turn == chess.BLACK:
        best_move = chess.Move(
            _flip_rank(best_canonical.from_square),
            _flip_rank(best_canonical.to_square),
            promotion=best_canonical.promotion,
        ).uci()
    else:
        best_move = best_canonical.uci()

    return scalar, wdl, moves_left, legal_moves, legal_lookup, policy_probs, best_move

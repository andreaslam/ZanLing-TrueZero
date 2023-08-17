import torch
import chess
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import network

board = chess.Board()

bigl = []

if torch.cuda.is_available():
    d = torch.device("cuda")
else:
    d = torch.device("cpu")

print("Using: " + str(d))


def convert_board(board, bigl):
    # FULL LIST HERE:
    # sq1 - white's turn
    # sq2 - black's turn
    # sq3, sq4 - castling pos l + r (us)
    # sq5, sq6 - castling pos l + r (opponent)
    # sql7, sql8 -  sqs for binary digits for the repetition counter
    # sq9 - sq20 - sqs for turn to move + non-turn to move's pieces
    # sq21 - en passant square if any

    # sq1 - white's turn
    # sq2 - black's turn
    if board.turn == chess.BLACK:
        sq1, sq2 = torch.zeros((8, 8)), torch.ones((8, 8))
        board = board.mirror()
    else:
        sq1, sq2 = torch.ones((8, 8)), torch.zeros((8, 8))

    # sq3, sq4 - castling pos l + r (us)
    # sq5, sq6 - castling pos l + r (opponent)

    # i think the castling is correct?

    us = board.turn
    opp = not us

    sq3, sq4 = torch.full(
        (8, 8), int(bool(board.castling_rights & chess.BB_A1))
    ), torch.full((8, 8), int(bool(board.castling_rights & chess.BB_H1)))

    sq5, sq6 = torch.full(
        (8, 8), int(bool(board.castling_rights & chess.BB_A8))
    ), torch.full((8, 8), int(bool(board.castling_rights & chess.BB_H8)))

    # 2 sqs for binary digits for the repetition counter

    rep = 2 if board.is_repetition(3) else 1 if board.is_repetition(2) else 0
    sq7 = torch.full((8, 8), rep)
    sq8 = torch.full((8, 8), board.halfmove_clock)

    # # pieces (ours)
    # all_bitboard_ours = []
    pieces = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    piece_sqs = []
    for colour in [chess.WHITE, chess.BLACK]:
        for piece in pieces:
            sq = torch.zeros(8, 8)
            for tile in board.pieces(piece, colour):
                sq[tile // 8, tile % 8] = 1
            piece_sqs.append(sq)

    # piece_sqs = torch.stack(piece_sqs)

    sq21 = torch.zeros((8, 8))

    all_data = [
        sq1,
        sq2,
        sq3,
        sq4,
        sq5,
        sq6,
        sq7,
        sq8,
        *piece_sqs,
        sq21,
    ]

    # Stack the tensors
    all_data = torch.stack(all_data)
    bigl.append(all_data)
    return all_data


with open("list.txt", "r") as f:
    contents = f.readlines()
    contents = [m.strip() for m in contents]


def eval_board(board, bigl):
    b = convert_board(board, bigl)
    try:
        model = torch.jit.load("tz.pt", map_location=d)
    except ValueError:
        model = torch.jit.script(network.TrueNet(num_resBlocks=2, num_hidden=128).to(d))
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    model.eval()
    with torch.no_grad():
        b = b.to(d)  # bring tensor to device
        # print(b.shape)
        policy, board_eval = model(b.unsqueeze(0))
        # print(policy)
        # print(board_eval)
        # logit_value, logit_win_pc, logit_draw_pc, logit_loss_pc, moves_left = (
        #     board_eval[0][0],
        #     board_eval[0][1],
        #     board_eval[0][2],
        #     board_eval[0][3],
        #     board_eval[0][4],
        # )
        # value = torch.tanh(logit_value).item()
        value = torch.tanh(board_eval).item()
        # l = F.softmax(board_eval[0][1:-1], dim=0)  # ignore board_eval and moves_left
        # logit_win_pc, logit_draw_pc, logit_loss_pc = (
        #     l[0].item(),
        #     l[1].item(),
        #     l[2].item(),
        # )
    mirrored = False
    if board.turn == chess.BLACK:
        board = board.mirror()
        mirrored = True
    policy = policy.tolist()
    # policy = policy[0]
    lookup = {}
    for p, c in zip(policy, contents):
        lookup[c] = p

    legal_lookup = {}
    legal_moves = list(board.legal_moves)

    for m in legal_moves:
        legal_lookup[str(m)] = lookup[str(m)]
    # softmax on policy
    sm = []
    for l in legal_lookup:
        sm.append(legal_lookup[l])

    sm = torch.tensor(sm)

    sm = F.softmax(sm, dim=0)

    sm = sm.tolist()

    for l, v in zip(legal_lookup, sm):
        legal_lookup[l] = v

    # print(move_counter)
    if (
        mirrored
    ):  # board.turn == chess.BLACK doesn't work since all the moves are in white's POV
        n = {}
        s = 0
        for move, key in zip(legal_lookup, legal_lookup.items()):
            n[move[0] + str(9 - int(move[1])) + move[2] + str(9 - int(move[3]))] = key[
                -1
            ]
            s += key[-1]
        legal_lookup = n
    # best_move = list(legal_lookup.keys())[0]

    # keep track of indices
    idx_li = []
    for move in legal_lookup:
        idx_li.append(contents.index(move))

    # print(idx_li)

    logit_win_pc, logit_draw_pc, logit_loss_pc = 0, 0, 0
    return value, logit_win_pc, logit_draw_pc, logit_loss_pc, legal_lookup, idx_li

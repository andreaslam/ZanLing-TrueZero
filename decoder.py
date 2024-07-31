import torch
import chess
import torch.nn.functional as F
import torch.nn as nnw
import torchvision
import network

board = chess.Board()

bigl = []

if torch.cuda.is_available():
    d = torch.device("cuda")
else:
    d = torch.device("cpu")

# # # print(("Using: " + str(d))


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
    is_mirrored = True
    if board.turn == chess.BLACK:
        sq1, sq2 = torch.zeros((8, 8)), torch.ones((8, 8))
        board = board.mirror()
    else:
        sq1, sq2 = torch.ones((8, 8)), torch.zeros((8, 8))

    # sq3, sq4 - castling pos l + r (us)
    # sq5, sq6 - castling pos l + r (opponent)

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
    if is_mirrored:
        board.mirror()
    # Stack the tensors
    all_data = torch.stack(all_data)

    bigl = torch.cat((bigl, all_data.unsqueeze(0)), dim=0)

    b, c, h, w = bigl.shape
    all_grid = torchvision.utils.make_grid(
        bigl.view(b * c, 1, h, w), nrow=c, padding=1, pad_value=0.3
    )
    torchvision.utils.save_image(all_grid, "BIGL.png")

    return all_data, bigl


with open("list.txt", "r") as f:
    contents = f.readlines()
    contents = [m.strip() for m in contents]


def eval_board(board, bigl):
    b = convert_board(board, bigl)

    model = torch.jit.load("tz.pt", map_location=d)
    model.to(d)
    model.eval()
    with torch.no_grad():
        b = b.to(d)  # bring tensor to device
        # # # # print((b.shape)
        policy, board_eval = model(b.unsqueeze(0))
        # # # print((board_eval)
        # # # # print((policy)
        # # # # print((board_eval)
        # logit_value, logit_win_pc, logit_draw_pc, logit_loss_pc, moves_left = (
        #     board_eval[0][0],
        #     board_eval[0][1],
        #     board_eval[0][2],
        #     board_eval[0][3],
        #     board_eval[0][4],
        # )
        # value = torch.tanh(logit_value).item()
        value = torch.tanh(torch.tensor(board_eval[0][0]))
        # l = F.softmax(board_eval[0][1:-1], dim=0)  # ignore board_eval and moves_left
        # logit_win_pc, logit_draw_pc, logit_loss_pc = (
        #     l[0].item(),
        #     l[1].item(),
        #     l[2].item(),
        # )

    mirrored = False
    if board.turn == chess.BLACK:
        # board = board.mirror()
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

    # # # # print((move_counter)
    if (
        board.turn == chess.BLACK
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

    # # # # print((legal_lookup)
    # board = board.mirror()
    logit_win_pc, logit_draw_pc, logit_loss_pc = 0, 0, 0
    return value, logit_win_pc, logit_draw_pc, logit_loss_pc, legal_lookup, idx_li


def decode_nn_output(board_eval, policy, board):
    # # print((list(board.legal_moves))
    with torch.no_grad():
        # # # print((policy)
        # # # print((board_eval)
        # logit_value, logit_win_pc, logit_draw_pc, logit_loss_pc, moves_left = (
        #     board_eval[0][0],
        #     board_eval[0][1],
        #     board_eval[0][2],
        #     board_eval[0][3],
        #     board_eval[0][4],
        # )
        # value = torch.tanh(logit_value).item()
        board_eval = board_eval.squeeze(0)

        board_eval = board_eval.tolist()

        value = torch.tanh(torch.tensor(board_eval[0]))

        legal_moves = list(board.legal_moves)
        fm = []
        if board.turn == chess.BLACK:
            # # print(("HI")
            value = -value
            legal_moves = list(board.legal_moves)
            for mv in legal_moves:
                mv = str(mv)
                m1, m3 = str(9 - int(mv[1])), str(9 - int(mv[3]))
                mv = mv[0] + m1 + mv[2] + m3
                fm.append(mv)
        else:
            fm = legal_moves

        legal_moves = fm

        # print((value)
        idx_li = []

        for mov in legal_moves:
            idx_li.append(contents.index(str(mov)))

        # print((idx_li)

        # step 2 - index all policy from legal moves

        pol_list = []
        for idx in idx_li:
            pol_list.append(policy.squeeze(0).tolist()[idx])

        sm = torch.tensor(pol_list)

        sm = F.softmax(sm, dim=0)

        sm = sm.tolist()

        # # # # print((move_counter)

        legal_moves = list(board.legal_moves)  # reverted back to normal

        legal_lookup = {}
        for mv, pol in zip(legal_moves, sm):
            legal_lookup[str(mv)] = pol

        # print((legal_lookup)
        best_move = sorted([(v, k) for k, v in legal_lookup.items()])[-1][1]

        # # print((idx_li)

        # re-flip legal moves again (before playing)

        logit_win_pc, logit_draw_pc, logit_loss_pc = 0, 0, 0
    return (
        value,
        logit_win_pc,
        logit_draw_pc,
        logit_loss_pc,
        legal_lookup,
        idx_li,
        best_move,
    )

# POLICY ONLY
import torch
import chess
import torch.nn.functional as nnf
import torchvision

# check for GPU

if torch.cuda.is_available():
    d = torch.device("cuda")
else:
    d = torch.device("cpu")

print("Using: " + str(d))

model = torch.jit.load("chess_16x128_gen3634.pt", map_location=d)
model.eval()

board = chess.Board()

# for now, we're (NN) playing as white

# order is probably pkbrkq
def convert_board(board, us, bigl):
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
        # print("BLACK")
        sq1, sq2 = torch.zeros((8,8)),torch.ones((8,8))
    else:
        # print("WHITE")
        sq1,sq2 = torch.ones((8,8)),torch.zeros((8,8))

    # sq3, sq4 - castling pos l + r (us)
    # sq5, sq6 - castling pos l + r (opponent)

    # i think the castling is correct?

    if us == 0:
        us = chess.WHITE
        opp = chess.BLACK
    else:
        us = chess.BLACK
        opp = chess.WHITE


    if us == chess.WHITE:
        sq3, sq4 = torch.full((8,8),int(bool(board.castling_rights & chess.BB_A1))) , torch.full((8,8),int(bool(board.castling_rights & chess.BB_H1)))
    else:
        sq3, sq4 = torch.full((8,8),int(bool(board.castling_rights & chess.BB_A8))) , torch.full((8,8),int(bool(board.castling_rights & chess.BB_H8)))

    if opp == chess.WHITE:
        sq5, sq6 = torch.full((8,8),int(bool(board.castling_rights & chess.BB_A1))) , torch.full((8,8),int(bool(board.castling_rights & chess.BB_H1)))
    else:
        sq5, sq6 = torch.full((8,8),int(bool(board.castling_rights & chess.BB_A8))) , torch.full((8,8),int(bool(board.castling_rights & chess.BB_H8)))

    # 2 sqs for binary digits for the repetition counter

    rep = 2 if board.is_repetition(3) else 1 if board.is_repetition(2) else 0
    sq7 = torch.full((8,8), rep)
    sq8 = torch.full((8,8), board.halfmove_clock)
    
    # pieces (ours)
    all_bitboard_ours = []
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    if board.turn == chess.WHITE: # white up
        use = 1
    else: # black up
        use = 0
    # if board.turn == chess.BLACK:
    #     board.apply_mirror()
    # determine which colour pieces to find
    # determine which colour affects what orientation
    for piece in pieces:
        bb = board.pieces(piece, use)
        
        bb = str(bb)
        
        if board.turn == chess.WHITE:
            bb = bb[::-1]
        
        bitboard = []

        b = []


        bb = bb.replace(".", "0")

        bb = bb.replace(" ", "")

        for x in bb:
            if x == "\n":
                bitboard.append(b)
                b = []
            else:
                b.append(int(x))
        bitboard.append(b) # don't forget about the last one!
        if board.turn == chess.WHITE:
            bx = []
            for x in bitboard:
                bx.append(x[::-1])
            bitboard = bx
            
        bitboard = torch.tensor(bitboard)
        all_bitboard_ours.append(bitboard)

    all_bitboard_ours = torch.stack(all_bitboard_ours)  # Use torch.stack() instead of torch.tensor()
    # if move_counter % 1: # since black would yield crap at the bottom
    #     board.apply_mirror()
    # pieces (opponent's)

    # determine which colour pieces to find
    # determine which colour affects what orientation
    
    all_bitboard_opps = [] # empty for now
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    if board.turn == chess.BLACK: # black down
        use = 1 
    else: # white down
        use = 0 
    # if board.turn == chess.WHITE:
    #     board.apply_mirror()
    for piece in pieces:
        bb = board.pieces(piece, use)
        bb = str(bb)
        
        if board.turn == chess.WHITE:
            bb = bb[::-1]
        
        bitboard = []
        

        b = []


        bb = bb.replace(".", "0")

        bb = bb.replace(" ", "")

        for x in bb:
            if x == "\n":
                bitboard.append(b)
                b = []
            else:
                b.append(int(x))
        bitboard.append(b) # don't forget about the last one!
        if board.turn == chess.WHITE:
            bx = []
            for x in bitboard:
                bx.append(x[::-1])
            bitboard = bx
        bitboard = torch.tensor(bitboard)

        all_bitboard_opps.append(bitboard)

    all_bitboard_opps = torch.stack(all_bitboard_opps)  # Use torch.stack() instead of torch.tensor()
    
    
    
    # sq21 - en passant square if any
    
    sq21 = torch.zeros((8,8))
    all_data = [sq1, sq2, sq3, sq4, sq5, sq6, sq7, sq8,
                all_bitboard_ours[0], all_bitboard_ours[1],
                all_bitboard_ours[2], all_bitboard_ours[3], all_bitboard_ours[4], all_bitboard_ours[5],
                all_bitboard_opps[0], all_bitboard_opps[1], all_bitboard_opps[2], all_bitboard_opps[3],
                all_bitboard_opps[4], all_bitboard_opps[5], sq21]

    # Stack the tensors
    all_data = torch.stack(all_data)
    bigl.append(all_data)
    return all_data, bigl

def eval_board(b, board, us):
    if us == 0:
        us = chess.WHITE
    else:
        us = chess.BLACK

    with torch.no_grad():
        b = b.to(d)  # bring tensor to device
        board_eval, policy = model(b.unsqueeze(0))
        # print(policy.shape)
        logit_value, logit_win_pc, logit_draw_pc, logit_loss_pc, moves_left = (
            board_eval[0][0],
            board_eval[0][1],
            board_eval[0][2],
            board_eval[0][3],
            board_eval[0][4],
        )
        value = torch.tanh(logit_value)
        l = nnf.softmax(board_eval[0][1:-1], dim=0)  # ignore board_eval and moves_left
        logit_win_pc, logit_draw_pc, logit_loss_pc = (
            l[0].item(),
            l[1].item(),
            l[2].item(),
        )
        # print(logit_win_pc, logit_draw_pc, logit_loss_pc)

    with open("list.txt", "r") as f:
        contents = f.readlines()
    contents = [m.strip() for m in contents]
    if board.turn == chess.BLACK:
        # n = []
        board.apply_mirror()
        # for m in contents:
        #     d1, d3 = str(9 - int(m[1])), str(9 - int(m[3]))
        #     m = m[0] + d1 + m[2] + d3
        #     n.append(m)
        # contents = n
    policy = policy.tolist()
    policy = policy[0]
    lookup = {}
    for p, c in zip(policy, contents):
        lookup[c] = p
    # print(lookup)
    legal_lookup = {}
    legal_moves = list(board.legal_moves)
    # print(legal_moves)
    for m in legal_moves:
        legal_lookup[str(m)] = lookup[str(m)]
    legal_lookup = dict(
        sorted(legal_lookup.items(), key=lambda item: item[1], reverse=True)
    )
    # print(legal_lookup)
    # print(board)
    best_move = list(legal_lookup.keys())[0]
    # print(legal_lookup)
    if move_counter % 2 == 1:
        board.apply_mirror()
        best_move = (
            best_move[0]
            + str(9 - int(best_move[1]))
            + best_move[2]
            + str(9 - int(best_move[3]))
        )
    return value, logit_win_pc, logit_draw_pc, logit_loss_pc, legal_lookup, best_move
selfplay = True
move_counter = 0 # ditching board.turn since idk how to use it
bigl = []
human = chess.WHITE
while not board.is_game_over():
    if selfplay:
        b, bigl = convert_board(board, move_counter % 2, bigl)
        # pass to NN
        value, logit_win_pc, logit_draw_pc, logit_loss_pc, legal_moves_policy_dict, best_move = eval_board(b, board, move_counter % 2)
        board.push(chess.Move.from_uci(best_move))
        print(best_move)
        move_counter += 1
    else:
        if board.turn == human:
            legal_moves = list(board.legal_moves)
            legal_moves = [str(x) for x in legal_moves]
            move = input(str(board) + "\n" + str(legal_moves) + ": ")
            board.push(chess.Move.from_uci(move))
            move_counter += 1
        else:
            b, bigl = convert_board(board, move_counter % 2, bigl)
            # pass to NN
            value, logit_win_pc, logit_draw_pc, logit_loss_pc, legal_moves_policy_dict, best_move = eval_board(b, board, move_counter % 2)
            board.push(chess.Move.from_uci(best_move))
            print(best_move)
            move_counter += 1

bigl = torch.stack(bigl, dim=0)
b, c, h, w = bigl.shape
all_grid = torchvision.utils.make_grid(
    bigl.view(b * c, 1, h, w),
    nrow=c, padding=1, pad_value=0.3
)
torchvision.utils.save_image(all_grid, "BIGL.png")
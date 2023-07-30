import torch
import chess
import torch.nn.functional as nnf
import torchvision
import math

# check for GPU

if torch.cuda.is_available():
    d = torch.device("cuda")
else:
    d = torch.device("cpu")

print("Using: " + str(d))

model = torch.jit.load("chess_16x128_gen3634.pt", map_location=d)
model.eval()


class Node:
    def __init__(self, board):
        self.parent = None  # Node
        self.children = []  # all legal children, as [Node, Node, Node ...]
        self.policy = None  # store its own policy
        self.visits = 0  # number of visits
        self.eval_score = 0  # initialise with no evalulation
        self.board = board
        self.total_action_value = 0  # total action value
        self.move_name = None  # chess.Move, eg e2e4

    def is_leaf(self):
        return self.visits == 0

    def get_q_val(self):  # child POV
        FPU = 0  # First Player Urgency
        q = self.total_action_value / self.visits if self.visits > 0 else FPU
        return q

    def puct_formula(self, parent_visits):  # child POV
        C_PUCT = 2  # "constant determining the level of exploration"
        u = C_PUCT * self.policy * (math.sqrt(parent_visits - 1)) / (1 + self.visits)
        q = self.get_q_val()
        result = q + u
        return result

    def is_terminal(self, board):
        return board.is_game_over()

    def __repr__(self) -> str:
        try:
            return self.move_name
        except Exception:
            return "Node at starting board position"

    def __str__(self) -> str:
        try:
            return "This is object of type Node and represents action " + str(
                self.move_name
            )
        except Exception:
            return "Node at starting board position"

    def eval_and_expand(self, board, move_counter):
        b = convert_board(board, move_counter % 2, bigl)
        (
            value,
            logit_win_pc,
            logit_draw_pc,
            logit_loss_pc,
            policy,
        ) = eval_board(b, board, move_counter % 2)
        self.eval_score = value
        self.visits = 0
        self.total_action_value = 0
        for child in policy:
            # create the child, append the list of child Node objects into self.children in the root node
            board.push(chess.Move.from_uci(child))
            cb = Node(board)
            cb.policy = policy[child]
            cb.parent = self
            cb.move_name = chess.Move.from_uci(child)
            print("child",cb)
            print("parent",cb.parent)
            # get q, PUCT (u) and then # q + u
            # print(upper_confidence_bound)
            self.children.append(cb)
            board.pop()  # remove the child move


class Tree:
    def __init__(self, board):
        self.board = board
        self.root_node = Node(board)

    def select(self):
        curr = self.root_node
        while curr.children:
            curr = max(curr.children, key=lambda n: n.puct_formula(curr.visits))
        return curr

    def backpropagate(self, node):
        # increment visit count
        n = node.eval_score
        while node:
            node.visits += 1
            # update action value w
            node.total_action_value += (
                n  # updated NN value incremented, not outdated evals
            )
            node = node.parent

    def step(self, move_counter):
        selected_node = self.select()
        if not selected_node.is_terminal(board):
            selected_node.eval_and_expand(board, move_counter)

        self.backpropagate(selected_node)

    def __repr__(self) -> str:
        try:
            return "This is object of type Node and represents action " + str(
                self.root_node.move_name
            )
        except Exception:
            return "Node at starting board position"

    def __str__(self) -> str:
        try:
            return "This is object of type Node and represents action " + str(
                self.root_node.move_name
            )
        except Exception:
            return "Node at starting board position"


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
        sq1, sq2 = torch.zeros((8, 8)), torch.ones((8, 8))
    else:
        # print("WHITE")
        sq1, sq2 = torch.ones((8, 8)), torch.zeros((8, 8))

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
        sq3, sq4 = torch.full(
            (8, 8), int(bool(board.castling_rights & chess.BB_A1))
        ), torch.full((8, 8), int(bool(board.castling_rights & chess.BB_H1)))
    else:
        sq3, sq4 = torch.full(
            (8, 8), int(bool(board.castling_rights & chess.BB_A8))
        ), torch.full((8, 8), int(bool(board.castling_rights & chess.BB_H8)))

    if opp == chess.WHITE:
        sq5, sq6 = torch.full(
            (8, 8), int(bool(board.castling_rights & chess.BB_A1))
        ), torch.full((8, 8), int(bool(board.castling_rights & chess.BB_H1)))
    else:
        sq5, sq6 = torch.full(
            (8, 8), int(bool(board.castling_rights & chess.BB_A8))
        ), torch.full((8, 8), int(bool(board.castling_rights & chess.BB_H8)))

    # 2 sqs for binary digits for the repetition counter

    rep = 2 if board.is_repetition(3) else 1 if board.is_repetition(2) else 0
    sq7 = torch.full((8, 8), rep)
    sq8 = torch.full((8, 8), board.halfmove_clock)

    # pieces (ours)
    all_bitboard_ours = []
    pieces = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    if board.turn == chess.WHITE:  # white up
        use = 1
    else:  # black up
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
        bitboard.append(b)  # don't forget about the last one!
        if board.turn == chess.WHITE:
            bx = []
            for x in bitboard:
                bx.append(x[::-1])
            bitboard = bx

        bitboard = torch.tensor(bitboard)
        all_bitboard_ours.append(bitboard)

    all_bitboard_ours = torch.stack(
        all_bitboard_ours
    )  # Use torch.stack() instead of torch.tensor()
    # if move_counter % 1: # since black would yield crap at the bottom
    #     board.apply_mirror()
    # pieces (opponent's)

    # determine which colour pieces to find
    # determine which colour affects what orientation

    all_bitboard_opps = []  # empty for now
    pieces = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    if board.turn == chess.BLACK:  # black down
        use = 1
    else:  # white down
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
        bitboard.append(b)  # don't forget about the last one!
        if board.turn == chess.WHITE:
            bx = []
            for x in bitboard:
                bx.append(x[::-1])
            bitboard = bx
        bitboard = torch.tensor(bitboard)

        all_bitboard_opps.append(bitboard)

    all_bitboard_opps = torch.stack(
        all_bitboard_opps
    )  # Use torch.stack() instead of torch.tensor()

    # sq21 - en passant square if any

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
        all_bitboard_ours[0],
        all_bitboard_ours[1],
        all_bitboard_ours[2],
        all_bitboard_ours[3],
        all_bitboard_ours[4],
        all_bitboard_ours[5],
        all_bitboard_opps[0],
        all_bitboard_opps[1],
        all_bitboard_opps[2],
        all_bitboard_opps[3],
        all_bitboard_opps[4],
        all_bitboard_opps[5],
        sq21,
    ]

    # Stack the tensors
    all_data = torch.stack(all_data)
    # .append(all_data)
    return all_data


with open("list.txt", "r") as f:
    contents = f.readlines()
    contents = [m.strip() for m in contents]


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

    if board.turn == chess.BLACK:
        board.apply_mirror()
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
    # print(move_counter)
    if (
        move_counter % 2 == 1
    ):  # board.turn == chess.BLACK doesn't work since all the moves are in white's POV
        board.apply_mirror()
        n = {}
        for move, key in zip(legal_lookup, legal_lookup.items()):
            n[move[0] + str(9 - int(move[1])) + move[2] + str(9 - int(move[3]))] = key[
                -1
            ]
        legal_lookup = n
    # best_move = list(legal_lookup.keys())[0]
    return value, logit_win_pc, logit_draw_pc, logit_loss_pc, legal_lookup


board = chess.Board()

tree = Tree(board)

bigl = []

move_counter = 0


MAX_NODES = 100

while not board.is_game_over():
    tree = Tree(board)
    while tree.root_node.visits < MAX_NODES:
        tree.step(move_counter)
    # select once
    best_move_node = max(tree.root_node.children, key=lambda n: n.visits)
    best_move = best_move_node.move_name
    print(best_move)
    board.push(best_move)
    move_counter += 1

# bigl = torch.stack(bigl, dim=0)
# b, c, h, w = bigl.shape
# all_grid = torchvision.utils.make_grid(
#     bigl.view(b * c, 1, h, w), nrow=c, padding=1, pad_value=0.3
# )
# torchvision.utils.save_image(all_grid, "BIGL.png")

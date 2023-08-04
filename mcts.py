import torch
import chess
import torchvision
import math
import decoder

class Node:
    def __init__(self, board, policy, parent, move_name):
        self.parent = parent  # Node
        self.children = []  # all legal children, as [Node, Node, Node ...]
        self.policy = policy  # store its own policy
        self.visits = 0  # number of visits
        self.eval_score = 0  # initialise with no evalulation
        self.board = board
        self.total_action_value = 0  # total action value
        self.move_name = move_name  # chess.Move, eg e2e4

    def is_leaf(self):
        return self.visits == 0

    def get_q_val(self):  # child POV
        FPU = 0  # First Player Urgency
        return (self.total_action_value / self.visits) if self.visits > 0 else FPU

    def puct_formula(self, parent_visits):  # child POV
        C_PUCT = 2  # "constant determining the level of exploration"
        u = C_PUCT * self.policy * (math.sqrt(parent_visits - 1)) / (1 + self.visits)
        q = self.get_q_val()
        result = q + u
        return result

    def is_terminal(self, board):
        return board.is_game_over()

    # def __repr__(self) -> str:
    #     if self.parent != None:
    #         u = 2 * self.policy * (math.sqrt(self.parent.visits - 1)) / (1 + self.visits)
    #         puct = self.puct_formula(self.parent.visits)
    #     else:
    #         u = "NaN"
    #         puct = "NaN"
    #     return ('Node(action="'
    #     + str(self.move_name)
    #     + '" V='
    #     + str(self.eval_score)
    #     + ", N="
    #     + str(self.visits)
    #     + ", W="
    #     + str(self.total_action_value)
    #     + ", P="
    #     + str(self.policy)
    #     + ", Q="
    #     + str(self.q)
    #     + ", U="
    #     + str(u)
    #     + ", PUCT="
    #     + str(puct)
    #     + ", len_children="
    #     + str(len(self.children))
    #     + ")")
    

    def __str__(self) -> str:
        if self.parent != None:
            u = 2 * self.policy * (math.sqrt(self.parent.visits - 1)) / (1 + self.visits)
            puct = self.puct_formula(self.parent.visits)
        else:
            u = "NaN"
            puct = "NaN"
        return ('Node(action="'
        + str(self.move_name)
        + '" V='
        + str(self.eval_score)
        + ", N="
        + str(self.visits)
        + ", W="
        + str(self.total_action_value)
        + ", P="
        + str(self.policy)
        + ", Q="
        + str(self.get_q_val())
        + ", U="
        + str(u)
        + ", PUCT="
        + str(puct)
        + ", len_children="
        + str(len(self.children))
        + ")")

    def layer_print(self, depth, MAX_TREE_PRINT_DEPTH):
        indent_count = depth + 2
        if depth <= MAX_TREE_PRINT_DEPTH: 
            if self.children:
                for c in self.children:
                    print("    " * indent_count ,c)
                    c.layer_print(depth+1, MAX_TREE_PRINT_DEPTH)

    def eval_and_expand(self, board, move_counter, bigl):
        # print(board)
        (
            value,
            logit_win_pc,
            logit_draw_pc,
            logit_loss_pc,
            policy,
            _best_move,
        ) = decoder.eval_board(board, bigl)
        print("    board FEN: " + board.fen())
        print("    ran NN:")
        print("         V=", str(value), "\n         policy=", str(policy))
        self.eval_score = value
        for p in policy:
            x = board.copy()
            x.push(chess.Move.from_uci(p))
            child = Node(board, policy[p], self, p)
            self.children.append(child)
            child.board = x

        # print("        children:",self.children)


class Tree:
    def __init__(self, board):
        self.board = board
        self.root_node = Node(board, None, None, None)

    def select(self):
        curr = self.root_node
        print("    selection:")
        while curr.children:
            curr = max(curr.children, key=lambda n: n.puct_formula(curr.visits))
            print("        ", curr)
            # print("        children:", curr.children)
        return curr

    def backpropagate(self, node):
        # increment visit count
        n = node.eval_score
        print("    backup:")
        while node:
            node.visits += 1
            # update action value w
            node.total_action_value += (
                n  # updated NN value incremented, not outdated evals
            )
            node = node.parent
            print("         updated node to " + str(node))
            n = -n

    def display_full_tree(self):
        # print("        root node:")
        # print("            ", self.root_node)
        # print("    children:")
        MAX_TREE_PRINT_DEPTH = 10
        print("    ", self.root_node)
        self.root_node.layer_print(0,MAX_TREE_PRINT_DEPTH)

    def step(self, move_counter, bigl):
        print("    root node:", self.root_node)

        selected_node = self.select()
        # self.display_full_tree()
        if not selected_node.is_terminal(board):
            bigl = selected_node.eval_and_expand(
                selected_node.board, move_counter, bigl
            )
            # self.display_full_tree()
        # print("        root node:", self.root_node)
        self.backpropagate(selected_node)
        self.display_full_tree()

    # utility function to display the entire tree

    def __str__(self) -> str:
        try:
            return "This is object of type Node and represents action " + str(
                self.root_node.move_name
            )
        except Exception:
            return "Node at starting board position"


with open("list.txt", "r") as f:
    contents = f.readlines()
    contents = [m.strip() for m in contents]


board = chess.Board()

# board.set_fen("r1b1k1nr/ppppb1pp/5pq1/4p3/2B1P2P/3PBQ2/PPP2PP1/RN3RK1 w kq - 2 10")

tree = Tree(board)

bigl = []

move_counter = 0


MAX_NODES = 10

while not board.is_game_over():
    tree = Tree(board)
    while tree.root_node.visits < MAX_NODES:
        print("step", tree.root_node.visits, ":")
        tree.step(move_counter, bigl)
    # select once
    best_move_node = max(tree.root_node.children, key=lambda n: n.visits)
    best_move = best_move_node.move_name
    print("bestmove", best_move)
    board.push(chess.Move.from_uci(best_move))
    move_counter += 1
    break

bigl = torch.stack(bigl, dim=0)
b, c, h, w = bigl.shape
all_grid = torchvision.utils.make_grid(
    bigl.view(b * c, 1, h, w), nrow=c, padding=1, pad_value=0.3
)
torchvision.utils.save_image(all_grid, "BIGL.png")

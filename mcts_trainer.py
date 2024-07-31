import torch
import chess
import torchvision
import math
import decoder
import torch
from torch.distributions.dirichlet import Dirichlet


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
        self.move_idx = None  # stores indexes for training

    def is_leaf(self):
        return self.visits == 0

    def get_q_val(self):  # child POV
        FPU = 0  # First Player Urgency
        return (self.total_action_value / self.visits) if self.visits > 0 else FPU

    def puct_formula(self, parent_visits):  # child POV
        C_PUCT = 2  # "constant determining the level of exploration"
        u = C_PUCT * self.policy * (math.sqrt(parent_visits - 1)) / (1 + self.visits)
        q = self.get_q_val()
        result = -q + u
        return result

    def is_terminal(self, board):
        return board.is_game_over()

    def __str__(self) -> str:
        if self.parent != None:
            u = (
                2
                * self.policy
                * (math.sqrt(self.parent.visits - 1))
                / (1 + self.visits)
            )
            puct = self.puct_formula(self.parent.visits)
        else:
            u = "NaN"
            puct = "NaN"
        return (
            'Node(action="'
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
            + ")"
        )

    def layer_p(self, depth, MAX_TREE_PRINT_DEPTH):
        indent_count = depth + 2
        if depth <= MAX_TREE_PRINT_DEPTH:
            if self.children:
                for c in self.children:
                    # print("    " * indent_count ,c)
                    c.layer_p(depth + 1, MAX_TREE_PRINT_DEPTH)

    def eval_and_expand(self, board, bigl):
        # # print(board)
        (
            value,
            logit_win_pc,
            logit_draw_pc,
            logit_loss_pc,
            policy,
            idx_li,
        ) = decoder.eval_board(board, bigl)
        # print("    board FEN: " + board.fen())
        # print("    ran NN:")
        # print("         V=", str(value), "\n         policy=", str(policy))
        self.eval_score = value
        for p in policy:
            x = board.copy()
            x.push(chess.Move.from_uci(p))
            child = Node(board, policy[p], self, p)
            self.children.append(child)
            child.board = x

        # # print("        children:",self.children)
        return idx_li


class Tree:
    def __init__(self, board):
        self.board = board
        self.root_node = Node(board, None, None, None)

    def select(self):
        curr = self.root_node
        # print("    selection:")
        while curr.children:
            curr = max(curr.children, key=lambda n: n.puct_formula(curr.visits))
            # print("        ", curr)
            # # print("        children:", curr.children)
        return curr

    def backpropagate(self, node):
        # increment visit count
        n = node.eval_score
        # print("    backup:")
        while node:
            node.visits += 1
            # update action value w
            node.total_action_value += (
                n  # updated NN value incremented, not outdated evals
            )
            node = node.parent
            # print("         updated node to " + str(node))
            n = -n

    def display_full_tree(self):
        # print("        root node:")
        # print("            ", self.root_node)
        # print("    children:")
        MAX_TREE_PRINT_DEPTH = float("inf")
        # print("    ", self.root_node)
        self.root_node.layer_p(0, MAX_TREE_PRINT_DEPTH)

    def step(self, bigl):
        # print("    root node:", self.root_node)
        EPS = 0.3  # 0.3 for chess
        selected_node = self.select()
        # self.display_full_tree()
        if not selected_node.is_terminal(selected_node.board):
            idx_li = selected_node.eval_and_expand(selected_node.board, bigl)
            # add Dirichlet noise if root node
            self.root_node.move_idx = idx_li
            if selected_node is self.root_node:
                noise = Dirichlet(
                    torch.tensor([0.3] * len(list(selected_node.board.legal_moves)))
                )
                samples = noise.sample()
                # print(len(list(selected_node.board.legal_moves)))
                for i in range(len(self.root_node.children)):
                    child = self.root_node.children[i]
                    child.policy = ((1 - EPS) * child.policy) + (EPS * samples[i])
            # self.display_full_tree()
        # print("        root node:", self.root_node)
        self.backpropagate(selected_node)
        # self.display_full_tree()

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


bigl = []

MAX_NODES = 10


def move(board):
    tree = Tree(board)
    while tree.root_node.visits < MAX_NODES:
        # # print("step", tree.root_node.visits, ":")
        tree.step(bigl)
    # select once
    best_move_node = max(tree.root_node.children, key=lambda n: n.visits)
    best_move = best_move_node.move_name
    # save pi (actual probability distribution)
    # get total visit counts first
    total_visits_list = [child.visits for child in tree.root_node.children]
    total_visits = sum(total_visits_list)
    pi = []  # vector containing actual visit probabilities after MCTS tree search
    for t in total_visits_list:
        pi.append(
            t / total_visits
        )  # probabilities should sum to 1, hence divide each probability by the total sum
    # call to store the root board input
    rb_input = decoder.convert_board(tree.root_node.board, [])
    memory_piece = rb_input
    return (
        best_move,
        memory_piece,
        pi,
        tree.root_node.move_idx,
    )  # return the best move, memory

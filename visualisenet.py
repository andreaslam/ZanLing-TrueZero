import torch
import matplotlib.pyplot as plt
import cv2
import imageio
import decoder
import chess
import os
import network


def extract_number(filename):
    return int(filename.split("tz_")[1].split(".")[0])


class ChessVisualizer:
    def __init__(
        self,
        model_path="tz_6515.pt",
        video_name="chess_game.mp4",
        gif_name="chess_game.gif",
        video_frame_size=(1600, 1200),
        video_fps=0.0075,
        i_d=0,
    ):
        self.gif_frames = []
        self.activations_startblock = []
        self.activations_backbone = []
        self.activations_policy = []
        self.activations_value = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.i_d = i_d
        self.architecture = (
            network.TrueNet(
                num_resBlocks=8,
                num_hidden=64,
                head_channel_policy=8,
                head_channel_values=4,
            )
            .to(self.device)
            .to(torch.float32)
        )
        self.architecture.load_state_dict(self.model.state_dict())
        self.video_name = video_name
        self.gif_name = gif_name
        self.video_frame_size = video_frame_size
        self.video_fps = video_fps

    def hook_startblock(self, module, input, output):
        self.activations_startblock.append(output)

    def hook_backbone(self, module, input, output):
        self.activations_backbone.append(output)

    def hook_policy(self, module, input, output):
        self.activations_policy.append(output)

    def hook_value(self, module, input, output):
        self.activations_value.append(output)

    def generate_gif(self):
        board = chess.Board()
        bigl = torch.tensor([])
        with torch.no_grad():
            while not board.is_game_over():
                input_data, bigl = decoder.convert_board(board, bigl)
                input_data = input_data.unsqueeze(0).to(self.device)
                self.activations_startblock = []
                self.activations_backbone = []
                self.activations_value = []
                self.activations_policy = []

                hook_handle_startblock = (
                    self.architecture.startBlock.register_forward_hook(
                        self.hook_startblock
                    )
                )
                hook_handle_policy = self.architecture.policyHead.register_forward_hook(
                    self.hook_policy
                )
                hook_handle_value = self.architecture.valueHead.register_forward_hook(
                    self.hook_value
                )
                hook_handles_backbone = [
                    self.architecture.backBone[i].register_forward_hook(
                        self.hook_backbone
                    )
                    for i in range(8)
                ]

                self.architecture(input_data)

                hook_handle_startblock.remove()
                for handle in hook_handles_backbone:
                    handle.remove()
                hook_handle_policy.remove()
                hook_handle_value.remove()

                plt.style.use("dark_background")
                fig = plt.figure(figsize=(8, 8))
                fig.suptitle("TrueZero Net Visualisation")
                fig.set_figheight(6)
                fig.set_figwidth(20)
                n_width = 1 + len(hook_handles_backbone) + 1

                plt.subplot(2, n_width, 1)
                plt.imshow(
                    self.activations_startblock[0][0]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(64, 64),
                    cmap="plasma",
                )
                plt.axis("off")
                plt.title("Start Block")

                for i, activation in enumerate(self.activations_backbone):
                    plt.subplot(2, n_width, i + 2)
                    plt.imshow(
                        activation[0, 0].clone().detach().cpu().numpy(), cmap="plasma"
                    )
                    plt.axis("off")
                    plt.title(f"Backbone {i}")

                for i in range(len(self.activations_value)):
                    plt.subplot(2, n_width, i + 10)
                    value, _, _, _, _, _, best_move = decoder.decode_nn_output(
                        self.activations_value[i], self.activations_policy[i], board
                    )
                    plt.imshow(
                        value.clone()
                        .detach()
                        .repeat(64)
                        .resize_(8, 8)
                        .detach()
                        .cpu()
                        .numpy(),
                        cmap="plasma",
                        vmin=-1,
                        vmax=1,
                    )
                    plt.axis("off")
                    plt.title("Value: " + str(round(value.item(), 5)))

                for i in range(len(self.activations_policy)):
                    # plt.ylim(0, 12) # TODO set fixed policy y axis, find max
                    plt.subplot(2, 1, 2)
                    _, _, _, _, legal_lookup, _, best_move = decoder.decode_nn_output(
                        self.activations_value[i], self.activations_policy[i], board
                    )
                    vals = torch.tensor(list(legal_lookup.values()))
                    labels = list(legal_lookup.keys())
                    num_moves = len(labels)
                    x = range(num_moves)
                    plt.bar(x, vals.detach().cpu().numpy())
                    plt.xticks(x, labels, rotation=45)
                    plt.title("Policy")

                board.push(chess.Move.from_uci(best_move))
                plt.savefig("temp_frame.png")
                plt.close()

                frame = cv2.cvtColor(cv2.imread("temp_frame.png"), cv2.COLOR_BGR2RGB)
                self.gif_frames.append(frame)

            imageio.mimsave(self.gif_name, self.gif_frames, duration=1 / self.video_fps)
            cv2.destroyAllWindows()

    def generate_evolution(self, max_pol, fen):
        # collect all nets first
        board = chess.Board()
        board.set_board_fen(fen)
        with torch.no_grad():
            input_data, _ = decoder.convert_board(board, torch.tensor([]))
            input_data = input_data.unsqueeze(0).to(self.device)
            self.activations_startblock = []
            self.activations_backbone = []
            self.activations_value = []
            self.activations_policy = []

            hook_handle_startblock = self.architecture.startBlock.register_forward_hook(
                self.hook_startblock
            )
            hook_handle_policy = self.architecture.policyHead.register_forward_hook(
                self.hook_policy
            )
            hook_handle_value = self.architecture.valueHead.register_forward_hook(
                self.hook_value
            )
            hook_handles_backbone = [
                self.architecture.backBone[i].register_forward_hook(self.hook_backbone)
                for i in range(8)
            ]

            self.architecture(input_data)

            hook_handle_startblock.remove()
            for handle in hook_handles_backbone:
                handle.remove()
            hook_handle_policy.remove()
            hook_handle_value.remove()

            plt.style.use("dark_background")
            fig = plt.figure(figsize=(8, 8))
            fig.suptitle(f"TrueZero visualisation: {net}")
            fig.set_figheight(6)
            fig.set_figwidth(20)
            n_width = 1 + len(hook_handles_backbone) + 1

            plt.subplot(2, n_width, 1)
            plt.imshow(
                self.activations_startblock[0][0]
                .detach()
                .cpu()
                .numpy()
                .reshape(64, 64),
                cmap="plasma",
            )
            plt.axis("off")
            plt.title("Start Block")

            for i, activation in enumerate(self.activations_backbone):
                plt.subplot(2, n_width, i + 2)
                plt.imshow(
                    activation[0, 0].clone().detach().cpu().numpy(), cmap="plasma"
                )
                plt.axis("off")
                plt.title(f"Backbone {i}")

            for i in range(len(self.activations_value)):
                plt.subplot(2, n_width, i + 10)
                value, _, _, _, _, _, best_move = decoder.decode_nn_output(
                    self.activations_value[i], self.activations_policy[i], board
                )
                plt.imshow(
                    value.clone()
                    .detach()
                    .repeat(64)
                    .resize_(8, 8)
                    .detach()
                    .cpu()
                    .numpy(),
                    cmap="plasma",
                    vmin=-1,
                    vmax=1,
                )
                plt.axis("off")
                plt.title("Value: " + str(round(value.item(), 5)))

            for i in range(len(self.activations_policy)):
                plt.subplot(2, 1, 2)
                plt.ylim(0, max_pol)  # set y-axis limits from 0 to 12
                _, _, _, _, legal_lookup, _, best_move = decoder.decode_nn_output(
                    self.activations_value[i], self.activations_policy[i], board
                )
                vals = torch.tensor(list(legal_lookup.values()))
                labels = list(legal_lookup.keys())
                num_moves = len(labels)
                x = range(num_moves)
                plt.bar(x, vals.detach().cpu().numpy())
                plt.xticks(x, labels, rotation=45)
                plt.title("Policy")

            plt.savefig(f"frames/temp_frame{self.i_d}.png")
            plt.close()

    def make_gif(self, paths):
        images = []

        def extract_number(filename):
            return int(filename.split("temp_frame")[1].split(".")[0])

        paths = sorted(paths, key=extract_number)
        for file in paths:
            frame = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
            images.append(frame)
        imageio.mimsave("evostartpos.gif", images, duration=1 / self.video_fps)
        cv2.destroyAllWindows()

    def get_max_policy(self, max_pol, fen):
        board = chess.Board()
        board.set_board_fen(fen)
        input_data, _ = decoder.convert_board(board, torch.tensor([]))
        input_data = input_data.unsqueeze(0).to(self.device)
        val, policy = self.model(input_data)
        value, _, _, _, legal_lookup, _, _ = decoder.decode_nn_output(
            val, policy, board
        )
        max_curr = max(list(legal_lookup.values()))
        if max_curr > max_pol:
            max_pol = max_curr
        return max_pol


if __name__ == "__main__":
    # visualizer = ChessVisualizer(model_path="tz_6515.pt")
    # visualizer.generate_gif()
    all_nets = []
    fen = "rnbqkbnr/pppp1p1p/6p1/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR"
    for net in os.listdir("nets"):
        all_nets.append("nets/" + net)

    all_nets = sorted(all_nets, key=extract_number)
    counter = 0
    max_pol = 0
    for net in all_nets:
        visualizer = ChessVisualizer(model_path=net, i_d=counter)
        max_pol = visualizer.get_max_policy(max_pol, fen)

    for net in all_nets:
        visualizer = ChessVisualizer(model_path=net, i_d=counter)

        visualizer.generate_evolution(max_pol, fen)
        counter += 1
    visualizer.make_gif(["frames/" + x for x in os.listdir("frames")])

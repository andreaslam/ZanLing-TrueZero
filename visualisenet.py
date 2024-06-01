import torch
import matplotlib.pyplot as plt
import cv2
import imageio
import chess
import os
import re
import argparse
from decoder import convert_board, decode_nn_output
from network import TrueNet

def extract_number(filename):
    return int(filename.split("tz_")[1].split(".")[0])

class ChessVisualiser:
    def __init__(self, model_path, video_name, gif_name, video_frame_size, video_fps, i_d):
        self.frames = []
        self.activations_startblock = []
        self.activations_backbone = []
        self.activations_policy = []
        self.activations_value = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.i_d = i_d
        self.architecture = TrueNet(num_resBlocks=8, num_hidden=64, head_channel_policy=8, head_channel_values=4)
        self.architecture.to(self.device).to(torch.float32)
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
                input_data, bigl = convert_board(board, bigl)
                input_data = input_data.unsqueeze(0).to(self.device)
                self._reset_activations()

                self._register_hooks()
                self.architecture(input_data)
                self._remove_hooks()

                fig = self._create_figure()
                board.push(self._process_activations(board, fig))
                self._save_frame(fig)
                self.frames.append(cv2.cvtColor(cv2.imread("temp_frame.png"), cv2.COLOR_BGR2RGB))

            imageio.mimsave(self.gif_name, self.frames, duration=1 / self.video_fps)
            cv2.destroyAllWindows()

    def generate_video(self):
        video_writer = cv2.VideoWriter(self.video_name, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, self.video_frame_size)
        
        for frame in self.frames:
            resized_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), self.video_frame_size)
            video_writer.write(resized_frame)
        
        video_writer.release()
        cv2.destroyAllWindows()

    def generate_evolution(self, max_pol, fen):
        board = chess.Board(fen)
        with torch.no_grad():
            input_data, _ = convert_board(board, torch.tensor([]))
            input_data = input_data.unsqueeze(0).to(self.device)
            self._reset_activations()

            self._register_hooks()
            self.architecture(input_data)
            self._remove_hooks()

            fig = self._create_figure(title=f"TrueZero visualisation: {self.i_d}")
            self._process_activations(board, fig, max_pol)
            plt.savefig(f"frames/temp_frame{self.i_d}.png")
            plt.close()

    def make_gif(self, paths):
        paths = sorted(paths, key=extract_number)
        images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in paths]
        imageio.mimsave("evostartpos.gif", images, duration=1 / self.video_fps)
        cv2.destroyAllWindows()

    def get_max_policy(self, max_pol, fen):
        board = chess.Board(fen)
        input_data, _ = convert_board(board, torch.tensor([]))
        input_data = input_data.unsqueeze(0).to(self.device)
        val, policy = self.model(input_data)
        _, _, _, _, legal_lookup, _, _ = decode_nn_output(val, policy, board)
        return max(max_pol, max(legal_lookup.values()))

    def _reset_activations(self):
        self.activations_startblock = []
        self.activations_backbone = []
        self.activations_policy = []
        self.activations_value = []

    def _register_hooks(self):
        self.hook_handles = [
            self.architecture.startBlock.register_forward_hook(self.hook_startblock),
            self.architecture.policyHead.register_forward_hook(self.hook_policy),
            self.architecture.valueHead.register_forward_hook(self.hook_value),
        ] + [self.architecture.backBone[i].register_forward_hook(self.hook_backbone) for i in range(8)]

    def _remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def _create_figure(self, title="TrueZero Net Visualisation"):
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle(title)
        fig.set_figheight(6)
        fig.set_figwidth(20)
        return fig

    def _process_activations(self, board, fig, max_pol=None):
        n_width = 1 + len(self.hook_handles) - 2

        plt.subplot(2, n_width, 1)
        plt.imshow(self.activations_startblock[0][0].cpu().numpy().reshape(64, 64), cmap="plasma")
        plt.axis("off")
        plt.title("Start Block")

        for i, activation in enumerate(self.activations_backbone):
            plt.subplot(2, n_width, i + 2)
            plt.imshow(activation[0, 0].cpu().numpy(), cmap="plasma")
            plt.axis("off")
            plt.title(f"Backbone {i}")

        for i, value in enumerate(self.activations_value):
            plt.subplot(2, n_width, i + 10)
            value, _, _, _, _, _, best_move = decode_nn_output(value, self.activations_policy[i], board)
            plt.imshow(value.cpu().numpy().repeat(64).reshape(8, 8), cmap="plasma", vmin=-1, vmax=1)
            plt.axis("off")
            plt.title("Value: " + str(round(value.item(), 5)))

        for i in range(len(self.activations_policy)):
            plt.subplot(2, 1, 2)
            if max_pol:
                plt.ylim(0, max_pol)
            _, _, _, _, legal_lookup, _, best_move = decode_nn_output(self.activations_value[i], self.activations_policy[i], board)
            vals = torch.tensor(list(legal_lookup.values()))
            labels = list(legal_lookup.keys())
            plt.bar(range(len(labels)), vals.cpu().numpy())
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.title("Policy")

        return chess.Move.from_uci(best_move)

    def _save_frame(self, fig):
        fig.savefig("temp_frame.png")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Chess Visualisation with Neural Networks")
    parser.add_argument("--mode", choices=["evolution", "play"], required=True, help="Mode to run: 'evolution' or 'play'")
    parser.add_argument("--nets_dir", default="nets", help="Directory containing neural network models")
    parser.add_argument("--output_dir", default="frames", help="Directory for output frames")
    parser.add_argument("--fen", default="rnbqkbnr/pppp1p1p/6p1/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR", help="FEN string for chess board")
    parser.add_argument("--video_name", default="chess_game.mp4", help="Output video name")
    parser.add_argument("--gif_name", default="chess_game.gif", help="Output GIF name")
    parser.add_argument("--video_frame_size", type=int, nargs=2, default=(2000, 600), help="Video frame size")
    parser.add_argument("--video_fps", type=float, default=30, help="Video FPS")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_nets = sorted([os.path.join(args.nets_dir, net) for net in os.listdir(args.nets_dir) if re.match(r"tz_\d+\.pt", net)], key=extract_number)
    visualiser = ChessVisualiser(model_path=all_nets[-1], video_name=args.video_name, gif_name=args.gif_name, video_frame_size=args.video_frame_size, video_fps=args.video_fps, i_d=0)
    
    if args.mode == "evolution":
        max_pol = 0
        for counter, net in enumerate(all_nets):
            visualiser = ChessVisualiser(model_path=net, video_name=args.video_name, gif_name=args.gif_name, video_frame_size=args.video_frame_size, video_fps=args.video_fps, i_d=counter)
            max_pol = visualiser.get_max_policy(max_pol, args.fen)
        
        for counter, net in enumerate(all_nets):
            visualiser = ChessVisualiser(model_path=net, video_name=args.video_name, gif_name=args.gif_name, video_frame_size=args.video_frame_size, video_fps=args.video_fps, i_d=counter)
            visualiser.generate_evolution(max_pol, args.fen)

        visualiser.make_gif([os.path.join(args.output_dir, x) for x in os.listdir(args.output_dir)])

    elif args.mode == "play":
        visualiser.generate_gif()
        visualiser.generate_video()

if __name__ == "__main__":
    main()

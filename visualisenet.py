import torch
import network
import matplotlib.pyplot as plt
import decoder
import chess
import cv2
import imageio

gif_frames = []

def hook_startblock(module, input, output):
    activations_startblock.append(output)

def hook_backbone(module, input, output):
    activations_backbone.append(output)
    
def hook_policy(module, input, output):
    activations_policy.append(output)

def hook_value(module, input, output):
    activations_value.append(output)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("mps")

# Load the model
model = torch.jit.load("./tz_6515.pt", map_location=device)
model.eval()
# Prepare input data
board = chess.Board()
# Create the network architecture
architecture = network.TrueNet(
    num_resBlocks=8, num_hidden=64, head_channel_policy=8, head_channel_values=4
).to(device).to(torch.float32)
architecture.load_state_dict(model.state_dict())

# Initialize video writer
video_name = 'chess_game.mp4'  # Change file extension to mp4
video_frame_size = (1600, 1200)  # Adjust frame size as needed
video_fps = 0.0075  # Frames per second


while not board.is_game_over():
    input_data = decoder.convert_board(board, []).unsqueeze(0).to(device)
    input_data = input_data.to(torch.float32)

    # Define hooks to collect activations
    activations_startblock = []
    activations_backbone = []
    activations_value = []
    activations_policy = []

    # Register hooks to collect activations
    hook_handle_startblock = architecture.startBlock.register_forward_hook(hook_startblock)
    hook_handle_policy = architecture.policyHead.register_forward_hook(hook_policy)
    hook_handle_value = architecture.valueHead.register_forward_hook(hook_value)
    hook_handles_backbone = [architecture.backBone[i].register_forward_hook(hook_backbone) for i in range(8)]

    # Forward pass through the network architecture
    with torch.no_grad():
        architecture.eval()
        architecture(input_data)

    # Unregister hooks to avoid accumulation
    hook_handle_startblock.remove()
    for handle in hook_handles_backbone:
        handle.remove()
    hook_handle_policy.remove()
    hook_handle_value.remove()

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8,8))  # Adjust the figure size
    fig.suptitle("TrueZero Net Visualisation")

    fig.set_figheight(6)
    fig.set_figwidth(20)

    n_width = 1 + len(hook_handles_backbone) + 1

    # Plot start block activations
    plt.subplot(2, n_width, 1)  
    plt.imshow(activations_startblock[0][0].detach().cpu().numpy().reshape(64, 64), cmap='plasma')
    plt.axis('off')
    plt.title('Start Block')

    # Plot backbone activations for each backbone
    for i, activation in enumerate(activations_backbone):
        plt.subplot(2, n_width, i+2)
        plt.imshow(activation[0, 0].clone().detach().cpu().numpy(), cmap='plasma')
        plt.axis('off')
        plt.title(f'Backbone {i}')

    # Plot the value activations
    for i in range(len(activations_value)):
        plt.subplot(2,n_width, i + 10)
        value, _, _, _, _, _, best_move = decoder.decode_nn_output(activations_value[i], activations_policy[i], board)
        plt.imshow(value.clone().detach().repeat(64).resize(8,8).detach().cpu().numpy(), cmap='plasma', vmin=-1, vmax=1)
        plt.axis('off')
        plt.title('Value: ' + str(round(value.item(),5)))

    # Plot the policy activations
    for i in range(len(activations_policy)):
        plt.subplot(2, 1, 2)
        
        _, _, _, _, legal_lookup, _, best_move = decoder.decode_nn_output(activations_value[i], activations_policy[i], board)
        vals = torch.tensor(list(legal_lookup.values()))
        labels = list(legal_lookup.keys())  # Extract labels for moves
        num_moves = len(labels)
        x = range(num_moves)  # Dynamically adjust x-axis based on the number of available moves
        plt.bar(x, vals.detach().cpu().numpy())
        plt.xticks(x, labels, rotation=45)  # Set the ticks and labels for the x-axis
        plt.title('Policy')

    print(best_move)
    
    board.push(chess.Move.from_uci(best_move))
    
    # Save the current plot as a frame in the video
    plt.savefig('temp_frame.png')  # Save the plot as an image
    plt.close()  # Close the plot to prevent it from being displayed

    # Read the saved image and write it to the video
    frame = cv2.cvtColor(cv2.imread('temp_frame.png'), cv2.COLOR_BGR2RGB)
    gif_frames.append(frame)  # Append frame to list

# Create GIF from frames
gif_name = 'chess_game.gif'
imageio.mimsave(gif_name, gif_frames, duration=1/video_fps)

# Release any resources and close windows
cv2.destroyAllWindows()
# Release the video writer and close any remaining windows

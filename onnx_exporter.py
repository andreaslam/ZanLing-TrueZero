import argparse
import network
import torch
import onnx
import sys


def main():
    parser = argparse.ArgumentParser(
        description="convert a PyTorch model to ONNX and display its input/output details."
    )
    parser.add_argument("model_path", type=str, help="path to the PyTorch model file.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="device to run the model on (default: auto-detect).",
    )

    args = parser.parse_args()

    try:
        device = torch.device(
            args.device
            if args.device
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {device}")

        # Initialize the model
        torch_model = network.TrueNet(
            num_resBlocks=8,
            num_hidden=64,
            head_channel_policy=8,
            head_channel_values=4,
        ).to(device)

        # Load the model state
        torch_model.load_state_dict(torch.jit.load(args.model_path).state_dict())

        # Create a dummy input tensor
        torch_input = torch.randn(1, 21, 8, 8).to(device)

        # Set the model to inference mode
        torch_model.eval()

        # Perform a forward pass
        torch_model(torch_input)

        # Export the model to ONNX format
        onnx_model_path = args.model_path.replace(".pt", ".onnx")
        torch.onnx.export(
            torch_model,
            torch_input,
            onnx_model_path,
            export_params=True,
            dynamic_axes={
                "input": {0: "batch_size"},  # variable length axes
                "output": {0: "batch_size"},
            },
            input_names=["input"],
            output_names=["output"],
        )
        print(f"Model exported to {onnx_model_path}")

        # Load the ONNX model
        model = onnx.load(onnx_model_path)

        inputs = {}
        for inp in model.graph.input:
            shape = str(inp.type.tensor_type.shape.dim)
            inputs[inp.name] = [int(s) for s in shape.split() if s.isdigit()]

        outputs = {}
        for out in model.graph.output:
            shape = str(out.type.tensor_type.shape.dim)
            outputs[out.name] = [int(s) for s in shape.split() if s.isdigit()]

        print("====================================")
        print("details about the neural network:")
        print("input name:", ", ".join(inputs.keys()))
        print("input shape:", ", ".join(str(v) for v in inputs.values()))
        print("output name:", ", ".join(outputs.keys()))
        print("output shape:", ", ".join(str(v) for v in outputs.values()))
        print("====================================")

    except Exception as e:
        print(f"Error: {e}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

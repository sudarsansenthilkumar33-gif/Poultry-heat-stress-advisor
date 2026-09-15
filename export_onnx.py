import torch
import torchvision.models as models

def export_mobilenet_to_onnx(output_path="mobilenet_v2_quant.onnx"):
    """Exports and optimizes PyTorch MobileNetV2 architecture for ONNX runtime."""
    print("Loading MobileNetV2 architecture...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()

    # Dummy input matching standard image dimensions [Batch, Channels, Height, Width]
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_tensor'],
        output_names=['output_logits'],
        dynamic_axes={'input_tensor': {0: 'batch_size'}, 'output_logits': {0: 'batch_size'}}
    )
    print("ONNX model exported successfully.")

if __name__ == '__main__':
    export_mobilenet_to_onnx()

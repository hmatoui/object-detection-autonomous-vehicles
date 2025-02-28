from ultralytics import YOLO
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def export_yolov8_to_onnx(model_path="yolov8s.pt", onnx_path="yolov8s.onnx"):
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Export the model to ONNX format
    model.export(format="onnx", dynamic=True, simplify=True)

    print(f"YOLOv8 model exported to {onnx_path}")


def onnx_to_tensorrt(onnx_file="yolov8s.onnx", engine_file="yolov8s.engine"):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Initialize TensorRT builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Load ONNX model
    with open(onnx_file, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(f"ERROR: {parser.get_error(error)}")
            return None

    # Set builder configuration
    config.max_workspace_size = 1 << 30  # 1GB of workspace
    builder.max_batch_size = 1
    config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 precision if supported

    # Build the TensorRT engine
    print("Building TensorRT engine...")
    engine = builder.build_engine(network, config)

    # Save the TensorRT engine to file
    with open(engine_file, "wb") as f:
        f.write(engine.serialize())

    print(f"TensorRT engine saved to {engine_file}")


def train_yolo():
    # Load the YOLOv8 model (pre-trained on COCO)
    model = YOLO("models/yolo/yolov8s.pt")  # Use "yolov8n.pt" for a smaller model

    # Train the model
    model.train(
        data="data/coco.yaml",   # Path to dataset configuration file
        epochs=50,              # Number of epochs
        batch=8,               # Batch size
        imgsz=320,              # Image size
        project="results",      # Directory to save results
        name="train_yolov8",    # Experiment name
        pretrained=True         # Use pre-trained weights
    )

if __name__ == "__main__":
    train_yolo()
    export_yolov8_to_onnx()
    onnx_to_tensorrt()

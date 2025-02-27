import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class YOLOv8TensorRT:
    def __init__(self, engine_file):
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load the TensorRT engine
        with open(engine_file, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate memory for inputs and outputs
        self.input_shape = (1, 3, 640, 640)
        self.input_size = trt.volume(self.input_shape) * trt.float32.itemsize
        self.output_shape = (1, 25200, 85)  # Modify based on YOLOv8 output
        self.output_size = trt.volume(self.output_shape) * trt.float32.itemsize

        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)

    def preprocess(self, frame):
        # Resize and normalize frame to (640, 640, 3)
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # Convert to CHW format
        img = np.expand_dims(img, axis=0)
        return img

    def infer(self, frame):
        # Preprocess the frame
        input_data = self.preprocess(frame)

        # Copy input data to GPU
        np.copyto(self.h_input, input_data.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output data back to CPU
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output.reshape(self.output_shape)

    def postprocess(self, output, frame):
        # Apply postprocessing like NMS and draw bounding boxes (custom implementation required)
        pass

def detect_realtime_tensorrt(engine_file, source=0):
    yolo_trt = YOLOv8TensorRT(engine_file)

    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        output = yolo_trt.infer(frame)

        # Postprocess and display results
        annotated_frame = yolo_trt.postprocess(output, frame)
        cv2.imshow("YOLOv8 TensorRT Real-Time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_realtime_tensorrt(engine_file="yolov8s.engine")

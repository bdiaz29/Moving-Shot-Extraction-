
import cv2
import copy
import numpy as np
import warnings
import onnx
import onnxruntime

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    # Ignore all DeprecationWarnings
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    trt.init_libnvinfer_plugins(None, '')

except Exception as ex:
    print("no trt installtion detected ")


class RaftOnnx():

    def __init__(self, model_path):
        # Initialize model
        self.initialize_model(model_path)

    def __call__(self, img1, img2):
        return self.estimate_flow(img1, img2)

    def initialize_model(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path,
                                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # Get model info
        self.get_input_details()
        self.get_output_details()
    def get_height(self):
        dim=self.input_height

        return self.input_height

    def get_width(self):
        dim=self.input_width
        return self.input_width

    def estimate_flow(self, img1, img2):
        input_tensor1 = self.prepare_input(img1)
        input_tensor2 = self.prepare_input(img2)

        outputs = self.inference(input_tensor1, input_tensor2)

        self.flow_map = self.process_output(outputs)

        return self.flow_map

    def prepare_input(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.img_height, self.img_width = img.shape[:2]

        img_input = cv2.resize(img, (self.input_width, self.input_height))

        # img_input = img_input/255
        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis, :, :, :]

        return img_input.astype(np.float32)

    def inference(self, input_tensor1, input_tensor2):
        # start = time.time()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor1,
                                                       self.input_names[1]: input_tensor2})

        # print(time.time() - start)
        return outputs

    def process_output(self, output):
        flow_map = output[1][0].transpose(1, 2, 0)

        return flow_map


    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        self.output_shape = model_outputs[0].shape
        self.output_height = self.output_shape[2]
        self.output_width = self.output_shape[3]

class Raft_TRT():
    def __init__(self, model_path):
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        with open(model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.get_input_details()
        self.get_output_details()
        self.output_shape = self.engine.get_binding_shape(1)
        self.output_height = self.output_shape[2]
        self.output_width = self.output_shape[3]
        self.outputshape = (2, self.input_height, self.input_width)
        p=0

    def get_height(self):
        return self.output_height

    def get_width(self):
        return self.output_width

    def __call__(self, img1, img2):
        return self.estimate_flow(img1, img2)

    def estimate_flow(self, img1, img2):
        input_tensor1 = self.prepare_input(copy.deepcopy(img1))
        input_tensor2 = self.prepare_input(copy.deepcopy(img2))
        outputs = self.inference(input_tensor1, input_tensor2)
        output = outputs[0]

        self.flow_map = self.process_output(output)
        outie = copy.deepcopy(self.flow_map)
        return outie

    def prepare_input(self, img):
        img=img[:, :, ::-1]
        self.img_height, self.img_width = img.shape[:2]
        img_input = cv2.resize(img, (self.input_width, self.input_height))
        # img_input = img_input/255
        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis, :, :, :]
        img_input = np.ascontiguousarray(img_input, dtype=np.float32)

        return img_input

    def inference(self, input_tensor1, input_tensor2):
        stream = cuda.Stream()
        inputs, outputs, bindings = self.allocate_buffers()

        # print("input_tensor1 shape:", input_tensor1.shape)
        # print("input_tensor1 dtype:", input_tensor1.dtype)

        cuda.memcpy_htod_async(bindings[0], input_tensor1, stream)
        cuda.memcpy_htod_async(bindings[1], input_tensor2, stream)
        self.context.execute_async(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0], bindings[2], stream)
        cuda.memcpy_dtoh_async(outputs[1], bindings[3], stream)
        stream.synchronize()
        return outputs

    def process_output(self, output):
        #flow_map = output.reshape((2, 240, 320))
        flow_map = output.reshape(self.outputshape)
        flow_map = flow_map.transpose(1, 2, 0)
        return flow_map


    def get_input_details(self):
        self.input_shape = self.engine.get_binding_shape(0)
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]


    def get_output_details(self):
        self.output_shape = self.engine.get_binding_shape(1)
        self.output_height = self.output_shape[2]
        self.output_width = self.output_shape[3]


    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(device_mem)
            if self.engine.binding_is_input(binding):
                inputs.append(host_mem)
            else:
                outputs.append(host_mem)
        return inputs, outputs, bindings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup TensorRT resources
        del self.context
        del self.engine


    def __del__(self):
        # Cleanup TensorRT resources
        del self.context
        del self.engine


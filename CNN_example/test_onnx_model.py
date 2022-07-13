import numpy as np
import onnxruntime as rt

model_file_path = "test_cnn.onnx"

# Open and print onnx model
# onnx_model = onnx.load(model_file_path)
# print(onnx_model)

# Test onnx model
sess = rt.InferenceSession(model_file_path)
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
batch_size = 1
dummy_input = np.random.random(
    (batch_size, *input_shape[1:])).astype(np.float32)
prediction = sess.run(None, {input_name: dummy_input})[0]
print(prediction)

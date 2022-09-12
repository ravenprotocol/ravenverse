import pickle as pkl

import numpy as np
import onnxruntime as rt


def download_onnx_model(model_path, onnx_model_name):
    test_model = pkl.load(open(model_path, "rb"))
    print("\n\n Pickle loaded model: \n")
    test_model.summary()
    test_model.get_onnx_model(onnx_model_name)


def test_onnx_model(onnx_model_path):
    sess = rt.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    batch_size = 1
    dummy_input = np.random.random(
        (batch_size, *input_shape[1:])).astype(np.float32)
    prediction = sess.run(None, {input_name: dummy_input})[0]
    print(prediction)

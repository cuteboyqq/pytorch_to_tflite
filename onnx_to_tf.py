from onnx_tf.backend import prepare
import onnx

onnx_model_path = 'yolov4_factory.onnx'
tf_model_path = 'yolov4_tf_factory'

onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)
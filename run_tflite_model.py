import numpy as np
import tensorflow as tf

tflite_model_path = 'yolov4_factory.tflite'
# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data
input_shape = input_details[0]['shape']

print('input_shape : {}'.format(input_shape))

input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()


YOLO_MODEL = True
if YOLO_MODEL:
    # get_tensor() returns a copy of the tensor data
    # use tensor() in order to get a pointer to the tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    print('output_data.shape : {}'.format(output_data.shape))
    
    output_data = interpreter.get_tensor(output_details[1]['index'])
    print(output_data)
    print('output_data.shape : {}'.format(output_data.shape))
    
    output_data = interpreter.get_tensor(output_details[2]['index'])
    print(output_data)
    print('output_data.shape : {}'.format(output_data.shape))
else:
    # get_tensor() returns a copy of the tensor data
    # use tensor() in order to get a pointer to the tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    print('output_data.shape : {}'.format(output_data.shape))
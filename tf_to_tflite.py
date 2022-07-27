import tensorflow as tf
import numpy as np


    
    
def demo():
  interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
  interpreter.allocate_tensors()
  #logging.info('tflite model loaded')

  input_details = interpreter.get_input_details()
  print(input_details)
  output_details = interpreter.get_output_details()
  print(output_details)

  input_shape = input_details[0]['shape']

  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

  print(output_data)
  
  
if __name__=="__main__":
    saved_model_dir = 'yolov4_tf_factory'
    tflite_model_path = 'yolov4_factory.tflite'

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Save the model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    demo()
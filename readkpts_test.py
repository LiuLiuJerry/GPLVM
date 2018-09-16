import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

model_dir = 'myNet/trained_models'
checkpoint_path = tf.train.latest_checkpoint(model_dir)
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(model_dir)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))

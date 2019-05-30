import numpy as np
import tensorflow as tf
from time import time
import math
from include.data import get_data_set
from include.model import model,lr
train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
tf.set_random_seed(21)
x, y, output, y_pred_cls, global_step, learning_rate = model
global_accuracy = 0
epoch_start = 0
import numpy as np
import tensorflow as tf
from time import time
import math
from include.data import get_data_set
from include.model import model,lr
train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
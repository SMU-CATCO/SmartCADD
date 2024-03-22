import numpy as np
import tensorflow as tf


def get_n_trainable_weights(model):
    return np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

import tensorflow as tf


def create_model(size, n_classes):
    inputs = tf.keras.layers.Input(shape=(15,))
    net = tf.keras.layers.Dense(size, activation="relu")(inputs)
    net = tf.keras.layers.Dense(n_classes, activation="softmax")(net)

    return tf.keras.models.Model(inputs, net)

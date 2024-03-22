import tensorflow as tf
from clay_project_template.layers import test_function


class DVector(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__()

        self.axis = axis

        data_format = "channels_first"
        if axis == 1:
            data_format = "channels_last"
        self.gap = tf.keras.layers.GlobalAveragePooling1D(
            data_format=data_format, name="gap"
        )

    def call(self, inputs):

        gap = self.gap(inputs)
        return gap

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
            }
        )
        return config

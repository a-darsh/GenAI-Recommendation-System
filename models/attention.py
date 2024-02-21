import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Permute, Multiply, Lambda, Softmax

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_weights = self.add_weight(name="att_weights",
                                                 shape=(input_shape[-1], 1),
                                                 initializer='random_normal',
                                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Apply attention weights
        attention_score = tf.matmul(inputs, self.attention_weights)
        attention_weights = Softmax(axis=1)(attention_score)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        base_config = super(AttentionLayer, self).get_config()
        return base_config

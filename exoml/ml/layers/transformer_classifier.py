import tensorflow as tf
from keras.layers import LeakyReLU

from exoml.ml.layers.dropout import AdaptiveStdDropout

try:
    from keras_nlp.layers import TransformerEncoder  # type: ignore
except ImportError:
    try:
        from keras_hub.layers import TransformerEncoder  # type: ignore
    except ImportError:

        @tf.keras.utils.register_keras_serializable()
        class TransformerEncoder(tf.keras.layers.Layer):
            def __init__(self, intermediate_dim, num_heads, dropout=0.0,
                         activation="relu", layer_norm_epsilon=1e-6,
                         kernel_initializer="glorot_uniform",
                         bias_initializer="zeros", normalize_first=False, **kwargs):
                super().__init__(**kwargs)
                self.intermediate_dim = intermediate_dim
                self.num_heads = num_heads
                self.dropout_rate = dropout
                self.activation = tf.keras.activations.get(activation)
                self.layer_norm_epsilon = layer_norm_epsilon
                self.kernel_initializer = kernel_initializer
                self.bias_initializer = bias_initializer
                self.normalize_first = normalize_first

            def build(self, input_shape):
                self.input_dim = input_shape[-1]
                self.att = tf.keras.layers.MultiHeadAttention(
                    num_heads=self.num_heads, key_dim=self.input_dim // self.num_heads,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                )
                self.ffn = tf.keras.Sequential([
                    tf.keras.layers.Dense(self.intermediate_dim,
                                          activation=self.activation,
                                          kernel_initializer=self.kernel_initializer,
                                          bias_initializer=self.bias_initializer),
                    tf.keras.layers.Dense(self.input_dim,
                                          kernel_initializer=self.kernel_initializer,
                                          bias_initializer=self.bias_initializer),
                ])
                self.att_norm = tf.keras.layers.LayerNormalization(
                    epsilon=self.layer_norm_epsilon)
                self.ffn_norm = tf.keras.layers.LayerNormalization(
                    epsilon=self.layer_norm_epsilon)
                self.att_dropout = tf.keras.layers.Dropout(self.dropout_rate)
                self.ffn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

            def call(self, inputs, mask=None, training=None):
                if self.normalize_first:
                    normed = self.att_norm(inputs)
                    attn_output = self.att(normed, normed, normed)
                    attn_output = self.att_dropout(attn_output)
                    out1 = inputs + attn_output
                    ffn_input = self.ffn_norm(out1)
                    ffn_output = self.ffn(ffn_input)
                    ffn_output = self.ffn_dropout(ffn_output)
                    return out1 + ffn_output
                else:
                    attn_output = self.att(inputs, inputs, inputs)
                    attn_output = self.att_dropout(attn_output)
                    out1 = self.att_norm(inputs + attn_output)
                    ffn_output = self.ffn(out1)
                    ffn_output = self.ffn_dropout(ffn_output)
                    return self.ffn_norm(out1 + ffn_output)

            def get_config(self):
                config = super().get_config()
                config.update({
                    "intermediate_dim": self.intermediate_dim,
                    "num_heads": self.num_heads,
                    "dropout": self.dropout_rate,
                    "activation": tf.keras.activations.serialize(self.activation),
                    "layer_norm_epsilon": self.layer_norm_epsilon,
                    "kernel_initializer": tf.keras.initializers.serialize(
                        tf.keras.initializers.get(self.kernel_initializer)),
                    "bias_initializer": tf.keras.initializers.serialize(
                        tf.keras.initializers.get(self.bias_initializer)),
                    "normalize_first": self.normalize_first,
                })
                return config


@tf.keras.utils.register_keras_serializable()
class ClassToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.layers.Layer):
    def __init__(self, num_heads, mlp_dim, dropout, layer_norm_epsilon, num_blocks, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.num_blocks = num_blocks

    def build(self, input_shape):
        self.transformer_encoders = []
        for _ in range(self.num_blocks):
            self.transformer_encoders.append(
                TransformerEncoder(
                    self.mlp_dim,
                    self.num_heads,
                    dropout=self.dropout,
                    activation=LeakyReLU(),
                    layer_norm_epsilon=self.layer_norm_epsilon,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    normalize_first=False
                )
            )

    def call(self, inputs):
        result = inputs
        for encoder in self.transformer_encoders:
            result = encoder(result)
        return result

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "num_blocks": self.num_blocks,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class TransformerClassifier(tf.keras.layers.Layer):
    def __init__(self, num_heads, mlp_dim, dropout, layer_norm_epsilon,
                 l1, l2, num_blocks, classes, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.l1 = l1
        self.l2 = l2
        self.num_blocks = num_blocks
        self.classes = classes

    def build(self, input_shape):
        self.input_proj = tf.keras.layers.Dense(
            self.mlp_dim, activation="relu", name="input_projection"
        )
        self.add_pos_embedding = AddPositionEmbs(name="add_pos_embedding")
        self.transformer = Transformer(
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            layer_norm_epsilon=self.layer_norm_epsilon,
            num_blocks=self.num_blocks,
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon, name="Transformer/encoder_norm"
        )
        activation = "softmax" if self.classes > 1 else "sigmoid"
        out_filters = self.classes if self.classes > 1 else 1
        self.head = tf.keras.layers.Dense(
            out_filters,
            activation=activation,
            name="head",
            activity_regularizer=tf.keras.regularizers.L1L2(
                l1=self.l1, l2=self.l2
            ),
        )
        if self.classes == 1:
            self.reshape = tf.keras.layers.Reshape((-1,), name="reshape_output")
        else:
            self.reshape = None

    def call(self, inputs):
        x = self.input_proj(inputs)
        x = self.add_pos_embedding(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        x = self.head(x)
        if self.reshape is not None:
            x = self.reshape(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "l1": self.l1,
            "l2": self.l2,
            "num_blocks": self.num_blocks,
            "classes": self.classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

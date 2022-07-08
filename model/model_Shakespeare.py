import tensorflow as tf


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim, batch_input_shape=[batch_size, None]
            ),
            tf.keras.layers.GRU(
                rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer="glorot_uniform",
            ),
            tf.keras.layers.Dense(vocab_size),
        ]
    )
    return model


def create_model_Shakespeare():
    model = build_model(
        vocab_size=65,  # no. of unique characters
        embedding_dim=256,  # 256
        rnn_units=1024,  # 1024
        batch_size=64,
    )

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )

    model.compile(optimizer="adam", loss=loss)

    return model

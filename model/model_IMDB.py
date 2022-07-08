import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text


def create_model_IMDB():
    bert_model_name = "small_bert/bert_en_uncased_L-2_H-128_A-2"

    map_name_to_handle = {
    "small_bert/bert_en_uncased_L-2_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1"}


    map_model_to_preprocess = {
    "small_bert/bert_en_uncased_L-2_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
}

    tfhub_handle_encoder = map_name_to_handle[bert_model_name]
    tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]
    def build_classifier_model():
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
        preprocessing_layer = hub.KerasLayer(
            tfhub_handle_preprocess, name="preprocessing"
        )
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(
            tfhub_handle_encoder, trainable=True, name="BERT_encoder"
        )
        outputs = encoder(encoder_inputs)
        net = outputs["pooled_output"]
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(net)
        return tf.keras.Model(text_input, net)

    classifier_model = build_classifier_model()

    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = tf.metrics.BinaryAccuracy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    return classifier_model,loss,optimizer,metrics

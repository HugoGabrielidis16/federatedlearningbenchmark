import tensorflow as tf


def create_model_CIFAR10():
    Resnet = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(32, 32, 3),
    )

    for layer in Resnet.layers:
        layer.trainable = False

    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = Resnet(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1024, activation="relu")(x)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)
    output = tf.keras.layers.Dense(units=10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    return model, loss, optimizer, metrics


if __name__ == "__main__":
    model, _, _, _ = create_model_CIFAR10()
    model.summary()

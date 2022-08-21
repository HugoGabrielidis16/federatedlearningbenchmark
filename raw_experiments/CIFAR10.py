import tensorflow as tf
from tensorflow.keras import layers

def create_model_CIFAR10():
    Resnet = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(32,32, 3)
    )

    for layer in Resnet.layers:
        layer.trainable = False
  
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    """
    x = tf.keras.Sequential([
      layers.RandomFlip("horizontal_and_vertical"),
      layers.RandomRotation(0.2),
        ])(inputs)

    """

    x = Resnet(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1024, activation="relu")(x)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)
    output = tf.keras.layers.Dense(units=10, activation="softmax")(x)

    model_CIFAR10 = tf.keras.Model(inputs=inputs, outputs=output)
    model_CIFAR10.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model_CIFAR10

def load_data_CIFAR10():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    def preprocess_image_input(input_images):
        input_images = input_images.astype("float32")
        output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
        return output_ims

    X_train = preprocess_image_input(X_train)
    X_test = preprocess_image_input(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
  model = create_model_CIFAR10()
  #model.summary()
  X_train, X_test, y_train, y_test = load_data_CIFAR10()
  model.fit(X_train,y_train, epochs = 100, validation_data= (X_test, y_test))


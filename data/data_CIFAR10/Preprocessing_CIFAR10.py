import tensorflow as tf

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
  X_train, X_test, y_train, y_test = load_data_CIFAR10()
  print(X_train.shape)
  print(X_test.shape)
  print(y_train.shape)
  print(y_test.shape)

  print(type(X_train))
  print(type(y_train))


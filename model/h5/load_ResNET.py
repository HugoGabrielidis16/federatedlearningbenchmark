import tensorflow as tf

if __name__ == "__main__":
  resnet = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3),
    )
  resnet.save("ResNET50.h5")
  

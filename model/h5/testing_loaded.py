import tensorflow as tf

if __name__ == "__main__":
  model = tf.keras.models.load_model("ResNET50.h5") 
  print(type(model))

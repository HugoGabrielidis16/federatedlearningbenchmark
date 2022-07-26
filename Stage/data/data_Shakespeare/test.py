import tensorflow as tf
import numpy as np

path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)

text = open(path_to_file, "rb").read()
text = text.decode(encoding="utf-8")
print(f"Length of text: {len(text)} characters")
vocab = sorted(set(text))
print(f"{len(vocab)} unique characters")

import tensorflow as tf
import numpy as np

path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)

text = open(path_to_file, "rb").read()
text = text.decode(encoding="utf-8")

vocab = sorted(set(text))

# Create a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
# Make a copy of the unique set elements in NumPy array format for later use in the decoding the predictions
idx2char = np.array(vocab)
# Vectorize the text with a for loop
text_as_int = np.array([char2idx[c] for c in text])


# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

print(text_as_int[0])
# for i in char_dataset.take(5):
#   print(i.numpy())
seq_length = 100  # The max. length for single input
# examples_per_epoch = len(text)//(seq_length+1) # double-slash for “floor” division
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

BUFFER_SIZE = 10000  # TF shuffles the data only within buffers

BATCH_SIZE = 64  # Batch size

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

print(list(dataset.as_numpy_iterator()))

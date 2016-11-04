import collections
import tensorflow as tf

class Reader(object):
  """
  The standard Reader class to read a word text file.
  It assumes a UTF-8 format and separates on whitespaces.
  """
  def __init__(self, filename, lang):
    self.filename = filename
    self.lang = lang
    self.words = []
    self.read_words()
    self.process_rare_words()
    self.build_vocab()
    self.file_to_word_ids()

  def read_words(self):
    with tf.gfile.GFile(self.filename, "r") as f:
      self.words = f.read().decode("utf-8").split()

  def process_rare_words(self):
    # Get a dictionary mapping words to frequency
    counter = collections.Counter(self.words)
    new_words = []
    for word in self.words:
      if counter[word] == 1:
        new_words.append(u'<UNK>')
      else:
        new_words.append(word)
    self.words = new_words

  def build_vocab(self):
    # Get a dictionary mapping words to frequency
    counter = collections.Counter(self.words)
    # Sort according to descending frequency
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    # Get list of rare words using this
    # rare = [x for x in count_pairs if x[1] <= 1]

    # Unzip the list to get words in sorted order of frequency
    words, _ = list(zip(*count_pairs))
    # Map words to their indices
    self.word_to_id = dict(zip(words, range(len(words))))

  def file_to_word_ids(self):
    self.data_ids = [self.word_to_id[word] for word in self.words if word in self.word_to_id]

  def produce(self, data_ids, batch_size, num_steps, name=None):
    with tf.name_scope(name, "Producer", [data_ids, batch_size, num_steps]):
      raw_data = tf.convert_to_tensor(data_ids, name="raw_data", dtype=tf.int32)

      data_len = tf.size(raw_data)
      batch_len = data_len // batch_size
      data = tf.reshape(raw_data[0 : batch_size * batch_len],
                        [batch_size, batch_len])

      epoch_size = (batch_len - 1) // num_steps
      assertion = tf.assert_positive(
          epoch_size,
          message="epoch_size == 0, decrease batch_size or num_steps")
      with tf.control_dependencies([assertion]):
        epoch_size = tf.identity(epoch_size, name="epoch_size")

      i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
      x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
      y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
      return x, y


def main():
  r = Reader("data/hindi.txt", "hi")


if __name__ == "__main__":
  main()
import codecs
import os
import collections
from six.moves import cPickle
import numpy as np


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, ipa_file, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding
        self.ipa_file = ipa_file

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.convert_ipa()
        self.create_batches()
        self.reset_batch_pointer()

    def convert_ipa(self):
        with codecs.open(self.ipa_file, "r", encoding=self.encoding) as f:
            data = f.readlines()
        table = {}
        vocab_size = 0
        for l in data:
            encoding = int(l.split('\t')[1][:-1])
            table[l.split('\t')[0]] = encoding
            if encoding > vocab_size:
                vocab_size = encoding
        table['.'] = vocab_size + 1
        table[' '] = vocab_size + 2
        table['<S>'] = vocab_size + 3
        table['</S>'] = vocab_size + 4
        table['\n'] = vocab_size + 5
        table['UNK'] = vocab_size + 6
        vocab_size += 7
        self.table = table
        self.ipa_vocab_size = vocab_size
        for x, value in np.ndenumerate(self.tensor):
            self.ipa_tensor[x] = table[self.reverse_vocab[value]]

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        counter.update(('<S>', '</S>', 'UNK'))  # add tokens for start end and unk
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reverse_vocab = dict(zip(range(len(self.chars)), self.chars))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, ['<S>'] + list(data) + ['</S>'])))
        self.ipa_tensor = np.copy(self.tensor)
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reverse_vocab = dict(zip(range(len(self.chars)), self.chars))
        self.tensor = np.load(tensor_file)
        self.ipa_tensor = np.copy(self.tensor)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        self.ipa_tensor = self.ipa_tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.ipa_tensor
        ydata = np.copy(self.ipa_tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

# The default loader for text
from .errors import FileNotFound
from .processors import default_process
from .strings import LOGS, FILES, ERRORS

from six.moves import cPickle

import codecs
import numpy as np
import os

class TextLoader(object):
    def __init__(self,
                 data_dir,
                 preprocess=None,
                 processor=default_process,
                 check_saved=True,
                 filename='input.txt',
                 encoding='utf-8'):
        self.input_file = input_file = os.path.join(data_dir, filename)
        self.vocab_file = vocab_file = os.path.join(data_dir, FILES[0])
        self.data_file = data_file = os.path.join(data_dir, FILES[1])
        self.ipa_vocab_file = ipa_vocab_file = os.path.join(data_dir, FILES[7])
        self.ipa_data_file = ipa_data_file = os.path.join(data_dir, FILES[8])
        self.splits_file = splits_file = os.path.join(data_dir, FILES[9])
        self.encoding = encoding
        if check_saved and \
           (os.path.exists(vocab_file) and os.path.exists(data_file) and \
            os.path.exists(ipa_vocab_file) and os.path.exists(ipa_data_file) and \
            os.path.exists(splits_file)):
            print(LOGS[1])
            self.load_processed()
            # No need to re-save processed data
        else:
            print(LOGS[0])
            if not os.path.exists(input_file):
                raise FileNotFound(ERRORS[1].format(path=input_file))
            # codecs is needed due to utf-8 setting
            with codecs.open(input_file, "r", encoding=encoding) as f:
                data = f.read()
            if preprocess is not None:
                data = preprocess(data)
            self.vocab, self.ipa_vocab, self.splits = processor(data)
            self.vocab_size = len(self.vocab)
            self.ipa_vocab_size = len(self.ipa_vocab)
            self.data = np.array(list(map(self.vocab.get, ['<S>'] + list(data) + ['</S>'])))
            # Generate the IPA vocabulary conditionally
            if self.ipa_vocab_size > 0:
                self.ipa_data = np.array(list(map(self.ipa_vocab.get, ['<S>'] + list(data) + ['</S>'])))
            else:
                self.ipa_data = np.copy(self.data)
            self.save_data()

    def load_processed(self):
        with open(self.vocab_file, 'rb') as f:
            self.vocab = cPickle.load(f)
        with open(self.ipa_vocab_file, 'rb') as f:
            self.ipa_vocab = cPickle.load(f)
        with open(self.splits_file, 'rb') as f:
            self.splits = cPickle.load(f)
        self.vocab_size = len(self.vocab)
        self.ipa_vocab_size = len(self.ipa_vocab)
        self.data = np.load(self.data_file)
        self.ipa_data = np.load(self.ipa_data_file)

    def save_data(self):
        np.save(self.data_file, self.data)
        np.save(self.ipa_data_file, self.ipa_data)
        with open(self.vocab_file, 'wb') as f:
            cPickle.dump(self.vocab, f)
        with open(self.ipa_vocab_file, 'wb') as f:
            cPickle.dump(self.ipa_vocab, f)
        with open(self.splits_file, 'wb') as f:
            cPickle.dump(self.splits, f)

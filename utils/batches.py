# Set of text processing APIs used in all models
from .error import NoBatchError
from .strings import ERRORS

import numpy as np

class BatchLoader(object):
    def __init__(self, input_seq, gen_output, batch_size, seq_length, extra_data=None):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_batches = int(input_seq.size / (self.batch_size * self.seq_length))
        # When the data (tensor) is too small
        if self.num_batches == 0:
            raise NoBatchError(ERRORS[0])
        xdata = input_seq[:self.num_batches * self.batch_size * self.seq_length]
        # output generated from the `gen_output` function passed into BatchLoader
        ydata = gen_output(xdata, extra_data)
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.pointer = 0

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

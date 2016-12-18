# Set of text processing APIs used in all models
from .errors import NoBatchError
from .strings import ERRORS

import numpy as np

class BatchLoader(object):
    def __init__(self, input_seq, generator, batch_size, seq_length, extra_data=None):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_batches = num_batches = int(input_seq.size / (self.batch_size * self.seq_length))
        # When the data (tensor) is too small
        if self.num_batches == 0:
            raise NoBatchError(ERRORS[0])
        xdata = input_seq[:self.num_batches * self.batch_size * self.seq_length]
        # output generated from the `gen_output` function passed into BatchLoader
        self.x_batches, self.y_batches = generator(xdata, batch_size, num_batches, extra_data)
        self.pointer = 0

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

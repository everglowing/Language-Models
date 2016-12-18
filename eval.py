from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import codecs
import importlib
import os
import utils.processors as processors

from config.models import models
from config.arguments import eval_parser
from six.moves import cPickle

from utils.batches import BatchLoader
from utils.errors import ModelNotFound
from utils.strings import ERRORS, LOGS, FILES
from utils.textloader import TextLoader

def main():
    args = eval_parser.parse_args()
    eval(args)

def eval(args):
    # Get saved args from trained model
    with open(os.path.join(args.save_dir, FILES[2]), 'rb') as f:
        saved_args = cPickle.load(f)

    model_config = models[saved_args.model]
    # used to process input text
    eval_processor = getattr(processors, model_config["eval_processor"])
    Model = getattr(importlib.import_module(model_config["module"]), "Model")
    # For evaluation, batch size has to be 1
    saved_args.batch_size = 1
    saved_args.seq_length = args.seq_length
    # Vocab is needed to convert text to numbers
    with open(os.path.join(args.save_dir, FILES[3]), 'rb') as f:
        vocab = cPickle.load(f)
    model = Model(saved_args, sample=False, evaluation=True)

    with codecs.open(args.text, 'r', encoding='utf-8') as f:
        text = f.read()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ModelNotFound(ERRORS[9])
        # calculate perplexity
        ppl = perplexity(sess, model, saved_args, text, vocab)
        print('perplexity: {0}'.format(ppl))

def perplexity(sess, model, saved_args, text, vocab):
    x, y = eval_processor(text, vocab, saved_args.back_steps, saved_args.seq_length)
    seq_length = saved_args.seq_length
    state = sess.run(model.initial_state)
    total_loss = 0.0
    for i in range(x.shape[0]):
        feed = {model.input_data: x[i:i+1, :], model.targets: y[i:i+1, :],
                model.initial_state: state}
        [state, loss] = sess.run([model.last_state, model.loss], feed)
        total_loss += loss.sum()
    # need to subtract off loss from padding tokens
    total_loss -= loss[total_len % seq_length - seq_length:].sum()
    avg_entropy = total_loss / len(text)
    return np.exp(avg_entropy)  # this is the perplexity

if __name__ == '__main__':
    main()

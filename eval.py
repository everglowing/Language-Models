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
    eval_processor = getattr(processors, model_config["eval_processor"]["function"])
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
        eval_extra = model_config["eval_processor"]["extra"]
        ppl = perplexity(sess, model, eval_extra, saved_args, text, vocab, eval_processor)
        print('perplexity: {0}'.format(ppl))

def perplexity(sess, model, model_config, saved_args, text, vocab, eval_processor):
    extra_data = build_extra_data(model_config, saved_args)
    x, y, total_len = eval_processor(text, vocab, saved_args.seq_length, extra_data)
    print(LOGS[4])
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

def build_extra_data(model_config, args):
    extra_data = {}
    for arg in model_config["extra_args"]:
        extra_data[arg] = getattr(args, arg)
    return extra_data

if __name__ == '__main__':
    main()

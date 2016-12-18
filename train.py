from __future__ import print_function

from six.moves import cPickle

from config.models import models
from config.arguments import parser
from utils.textloader import TextLoader
from utils.batches import BatchLoader
from utils.strings import ERRORS

import utils.generators as generators
import utils.processors as processors

import numpy as np
import tensorflow as tf

import importlib
import os
import time

def main():
    args = parser.parse_args()
    train(args)

def train(args):
    model_config = models[args.model]
    generator = getattr(generators, model_config["generator"])
    processor = getattr(processors, model_config["processor"])
    Model = getattr(importlib.import_module(model_config["module"]), "Model")

    data_loader = TextLoader(data_dir=args.data_dir,
                             processor=processor)
    batch_loader = BatchLoader(input_seq=data_loader.data,
                               gen_output=generator,
                               batch_size=args.batch_size,
                               seq_length=args.seq_length)

    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        check_init_from(init=args.init_from, ckpt=ckpt)

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl')) as f:
            saved_model_args = cPickle.load(f)
        check_saved_args(saved_model_args, args)

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl')) as f:
            saved_vocab = cPickle.load(f)
        assert saved_vocab==data_loader.vocab, ERRORS[8]

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump(data_loader.vocab, f)

    model = Model(args)

    with tf.Session() as sess:
        writer = tf.train.SummaryWriter("logs", sess.graph)
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                if b == 0:
                    feed = {model.input_data: x, model.targets: y}
                else:
                    feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                train_loss, state, _ = sess.run([model.cost, model.last_state, model.train_op], feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start))
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                    or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

def check_init_from(init="", ckpt=None):
    assert os.path.isdir(args.init_from), ERRORS[2].format(init=init)
    assert os.path.isfile(os.path.join(args.init_from,"config.pkl")), ERRORS[3].format(init=init)
    assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")), ERRORS[4].format(init=init)
    assert ckpt, ERRORS[5]
    assert ckpt.model_checkpoint_path, ERRORS[6]

def check_saved_args(saved_model_args, args):
    need_be_same=["cell", "rnn_size", "num_layers", "seq_length"]
    for arg in need_be_same:
        assert vars(saved_model_args)[arg]==vars(args)[arg], ERRORS[7].format(arg=arg)


if __name__ == '__main__':
    main()

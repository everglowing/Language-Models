from __future__ import print_function

from six.moves import cPickle

from config.models import models
from config.arguments import parser
from utils.textloader import TextLoader
from utils.batches import BatchLoader
from utils.strings import ERRORS, LOGS, FILES

import utils.generators as generators
import utils.processors as processors

import numpy as np
import tensorflow as tf

import importlib
import json
import os
import time

def main():
    args = parser.parse_args()
    train(args)

def train(args):
    model_config = models[args.model]
    generator = getattr(generators, model_config["generator"]["function"])
    processor = getattr(processors, model_config["processor"])
    Model = getattr(importlib.import_module(model_config["module"]), "Model")

    data_loader = TextLoader(data_dir=args.data_dir,
                             processor=processor)
    extra_data = build_extra_data(model_config["generator"]["extra"], args, data_loader)
    batch_loader = BatchLoader(input_seq=data_loader.data,
                               generator=generator,
                               batch_size=args.batch_size,
                               seq_length=args.seq_length,
                               extra_data=extra_data)

    args.vocab_size = data_loader.vocab_size
    args.ipa_vocab_size = data_loader.ipa_vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        check_init_from(init=args.init_from, ckpt=ckpt)

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, FILES[2])) as f:
            saved_model_args = cPickle.load(f)
        check_saved_args(saved_model_args, args)

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, FILES[3])) as f:
            saved_vocab = cPickle.load(f)
        assert saved_vocab == data_loader.vocab, ERRORS[8]
        with open(os.path.join(args.init_from, FILES[7])) as f:
            saved_ipa_vocab = cPickle.load(f)
        assert saved_ipa_vocab == data_loader.ipa_vocab, ERRORS[8]

    with open(os.path.join(args.save_dir, FILES[2]), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, FILES[3]), 'wb') as f:
        cPickle.dump(data_loader.vocab, f)
    with open(os.path.join(args.save_dir, FILES[7]), 'wb') as f:
        cPickle.dump(data_loader.ipa_vocab, f)
    # Writing a textual summary of the model being used
    with open(os.path.join(args.save_dir, FILES[4]), 'wb') as f:
        summary = model_config["summary"] + "\n" + json.dumps(args.__dict__)
        f.write(summary)
    # Empty plotting file
    with open(os.path.join(args.save_dir, FILES[5]), 'wb') as f:
        f.write("")

    model = Model(args)

    with tf.Session() as sess:
        writer = tf.train.SummaryWriter("logs", sess.graph)
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        saver2 = tf.train.Saver(build_variables(model_config["build_variables"]))
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        if args.init_weights is not None:
            weight_model_path = tf.train.get_checkpoint_state(args.init_weights).model_checkpoint_path
            saver2.restore(sess, weight_model_path)
            print(LOGS[6])

        # Actual training starts now
        plot_data = ""
        for e in range(args.num_epochs):
            run_epoch(sess, model, saver, saver2, args, batch_loader, e, plot_data)

def run_epoch(sess, model, saver, saver2, args, batch_loader, e, plot_data):
    # Start off by tuning the correct learning rate for the epoch
    sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
    # Reset batch pointer back to zero
    batch_loader.reset_batch_pointer()
    # Start from an empty RNN state
    state = sess.run(model.initial_state)
    for b in range(batch_loader.num_batches):
        start = time.time()
        x, y = batch_loader.next_batch()
        if b == 0:
            feed = {model.input_data: x, model.targets: y}
        else:
            # Alternate approach, need to check
            # feed = {model.input_data: x, model.targets: y}
            # for i, (c, h) in enumerate(model.initial_state):
            #     feed[c] = state[i].c
            #     feed[h] = state[i].h
            feed = {model.input_data: x, model.targets: y, model.initial_state: state}
        train_loss, state, _ = sess.run([model.cost, model.last_state, model.train_op], feed)
        end = time.time()
        # print the result so far on terminal
        batch_num = e * batch_loader.num_batches + b
        total_num = args.num_epochs * batch_loader.num_batches
        print(LOGS[2].format(batch_num, total_num, e, train_loss, end - start))

        # Append the string plot_data to see trends
        plot_data += str(batch_num) + "," + str(train_loss) + "\n"

        # Save after `args.save_every` batches or at the very end
        if batch_num % args.save_every == 0 or \
           (e == args.num_epochs-1 and b == batch_loader.num_batches-1):
            checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
            weights_path = os.path.join(args.weights_dir, 'model_weights.ckpt')
            saver.save(sess, checkpoint_path, global_step=batch_num)
            saver2.save(sess, weights_path, global_step=batch_num)
            print(LOGS[3].format(checkpoint_path))
            # Plot loss vs batches on training data
            with open(os.path.join(args.save_dir, FILES[5]), 'a') as f:
                f.write(plot_data)
                plot_data = ""


def check_init_from(init="", ckpt=None):
    assert os.path.isdir(args.init_from), ERRORS[2].format(init=init)
    assert os.path.isfile(os.path.join(args.init_from, FILES[2])), ERRORS[3].format(init=init)
    assert os.path.isfile(os.path.join(args.init_from, FILES[3])), ERRORS[4].format(init)
    assert ckpt, ERRORS[5]
    assert ckpt.model_checkpoint_path, ERRORS[6]

def check_saved_args(saved_model_args, args):
    need_be_same=["cell", "rnn_size", "num_layers", "seq_length"]
    for arg in need_be_same:
        assert vars(saved_model_args)[arg]==vars(args)[arg], ERRORS[7].format(arg=arg)

def build_extra_data(model_config, args, data_loader):
    extra_data = {}
    for arg in model_config["extra_args"]:
        extra_data[arg] = getattr(args, arg)
    for d in model_config["data_loader"]:
        extra_data[d] = getattr(data_loader, d)
    return extra_data

def build_variables(build_vars):
    var_list = []
    for var in tf.all_variables():
        if var.name.startswith(build_vars):
            var_list.append(var)
    # empty list indicates whole model
    if len(var_list) == 0:
        var_list = tf.all_variables()
    return var_list


if __name__ == '__main__':
    main()

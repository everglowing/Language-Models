# Parse all the default arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data',
                   help='data directory containing input.txt')
parser.add_argument('--save_dir', type=str, default='save',
                   help='directory to store checkpointed models')
parser.add_argument('--rnn_size', type=int, default=128,
                   help='size of RNN hidden state')
parser.add_argument('--num_layers', type=int, default=2,
                   help='number of layers in the RNN')
parser.add_argument('--cell', type=str, default='lstm',
                   help='rnn, gru, or lstm')
parser.add_argument('--dropout', type=bool, default=False,
                   help='Presence of Dropout')
parser.add_argument('--keep_prob', type=float, default=0.8,
                   help='Keep Probability of dropout')
parser.add_argument('--batch_size', type=int, default=50,
                   help='minibatch size')
parser.add_argument('--seq_length', type=int, default=50,
                   help='RNN sequence length')
parser.add_argument('--num_epochs', type=int, default=50,
                   help='number of epochs')
parser.add_argument('--save_every', type=int, default=1000,
                   help='save frequency')
parser.add_argument('--grad_clip', type=float, default=5.,
                   help='clip gradients at this value')
parser.add_argument('--learning_rate', type=float, default=0.002,
                   help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.97,
                   help='decay rate for rmsprop')
parser.add_argument('--init_from', type=str, default=None,
                   help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'config.pkl'        : configuration;
                        'chars_vocab.pkl'   : vocabulary definitions;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)

subparsers = parser.add_subparsers(help='Type of model involved', dest='model')
a_parser = subparsers.add_parser("partial_brnn")

a_parser.add_argument('--back_steps', type=int, default=5,
                     help='Number of steps in BRNN')

eval_parser = argparse.ArgumentParser()
eval_parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
eval_parser.add_argument('--text', type=str,
                        help='filename of text to evaluate on')
eval_parser.add_argument('--seq_length', type=str, default=200,
                        help='Length of evaluation sequence length')
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import codecs
import numpy as np

class Model():
    def __init__(self, args, infer=False, evaluation=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size, state_is_tuple=True)
        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)
        if not evaluation and args.dropout == True:
            print "Using dropout layer"
            self.cell = cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=args.keep_prob)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm_ipa'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.ipa_vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.ipa_vocab_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.ipa_vocab_size, args.rnn_size])
                inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        outputs = []
        state = self.initial_state

        with tf.variable_scope("RNN_ipa"):
            for time_step in range(args.seq_length):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                if time_step == args.seq_length - 1:
                    self.final_state = state
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b

        loss_len = args.ipa_vocab_size

        self.probs = tf.nn.softmax(self.logits)
        self.loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                loss_len)
        self.cost = tf.reduce_sum(self.loss) / args.batch_size / args.seq_length
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def convert_ipa(self, tensor, vocab):
        reverse_vocab = {v: k for k, v in vocab.iteritems()}
        with codecs.open(self.args.ipa_file, "r", encoding='utf-8') as f:
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
        ipa_tensor = tensor[:]
        #ipa_tensor = table[reverse_vocab[tensor]]
        for x, value in enumerate(tensor):
            ipa_tensor[x] = table[reverse_vocab[value]]
        return ipa_tensor

    def eval(self, sess, chars, vocab, text):
        batch_size = 200
        state = sess.run(self.cell.zero_state(1, tf.float32))
        x = [vocab[c] if c in vocab else vocab['UNK'] for c in text]
        x = [vocab['<S>']] + x + [vocab['</S>']]
        total_len = len(x) - 1
        # pad x so the batch_size divides it
        while len(x) % 200 != 1:
            x.append(vocab[' '])
        ipa_x = self.convert_ipa(x, vocab)
        y = np.array(ipa_x[1:]).reshape((-1, batch_size))
        ipa_x = np.array(ipa_x[:-1]).reshape((-1, batch_size))
        print "Data loaded"
        total_loss = 0.0
        for i in range(ipa_x.shape[0]):
            feed = {self.input_data: ipa_x[i:i+1, :], self.targets: y[i:i+1, :],
                    self.initial_state: state}
            [state, loss] = sess.run([self.final_state, self.loss], feed)
            total_loss += loss.sum()
        # need to subtract off loss from padding tokens
        total_loss -= loss[total_len % batch_size - batch_size:].sum()
        avg_entropy = total_loss / len(text)
        return np.exp(avg_entropy)  # this is the perplexity

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret



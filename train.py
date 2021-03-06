#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import data_helpers_nsmc
from text_cnn import TextCNN
from tensorflow.contrib import learn

import sys

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Data loading params for nsmc only
tf.flags.DEFINE_string("pos_train_data_file", "./data/nsmc/train/nsmc_pos_train.txt", "Data source for the positive training data.")
tf.flags.DEFINE_string("pos_validate_data_file", "./data/nsmc/validate/nsmc_pos_validate.txt", "Data source for the positive validation data.")
tf.flags.DEFINE_string("neg_train_data_file", "./data/nsmc/train/nsmc_neg_train.txt", "Data source for the negative training data.")
tf.flags.DEFINE_string("neg_validate_data_file", "./data/nsmc/validate/nsmc_neg_validate.txt", "Data source for the negative validation data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate (default: 1e-3)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

def extract_dict(vocab_proc, dict_path):
    if not vocab_proc or not dict_path:
        print("[extract_dict] [ERROR] invalid argument passed!!!")
        return
    # reference: https://stackoverflow.com/questions/40661684/tensorflow-vocabularyprocessor
    # Extract word:id mapping from the object
    vocab_dict = vocab_proc.vocabulary_._mapping
    # Sort the vocabulary dictionary on the basis of values(id)
    sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
    vocabulary = list(list(zip(*sorted_vocab))[0])
    print("[extract_dict] length of vocabulary: " + str(len(vocabulary)))
    outf = open(dict_path + "/extracted_vocab.txt", "wt")
    for token in vocabulary:
        outf.write(token.strip() + "\n")
    outf.flush()
    outf.close()
    print("[extract_dict] vocabulary extracted into a text file.")

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    # extract_dict(vocab_processor)

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev

# x_train, y_train, vocab_processor, x_dev, y_dev = preprocess_nsmc_data()
def preprocess_nsmc_data(is_baseline):
    # Data Preparation for nsmc
    # ==================================================

    # Load data
    print("Loading nsmc data...")
    pos_train_examples, pos_val_examples, neg_train_examples, neg_val_examples, max_len =\
        data_helpers_nsmc.load_nsmc_train_val_data(
            FLAGS.pos_train_data_file, FLAGS.pos_validate_data_file,
            FLAGS.neg_train_data_file, FLAGS.neg_validate_data_file,
            not is_baseline)  # baseline or test model

    # Build vocabulary
    # max_document_length = max([len(x.split(" ")) for x in x_text])
    print("[neo] maximum corpus string length: " + str(max_len))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_len)
    x_examples = pos_train_examples + neg_train_examples + pos_val_examples + neg_val_examples
    x = np.array(list(vocab_processor.fit_transform(x_examples)))
    y = np.array([[0, 1] for _ in range(len(pos_train_examples))] +\
        [[1, 0] for _ in range(len(neg_train_examples))] +\
        [[0, 1] for _ in range(len(pos_val_examples))] +\
        [[1, 0] for _ in range(len(neg_val_examples))])
    # extract_dict(vocab_processor)

    # Randomly shuffle training data only
    np.random.seed(10)
    train_size = len(pos_train_examples) + len(neg_train_examples)
    val_size = len(pos_val_examples) + len(neg_val_examples)
    shuffle_indices = np.random.permutation(np.arange(train_size))
    x[:train_size] = x[shuffle_indices]
    y[:train_size] = y[shuffle_indices]

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(train_size, val_size))

    del pos_train_examples, pos_val_examples, neg_train_examples, neg_val_examples, x_examples, shuffle_indices

    return x[:train_size], y[:train_size], vocab_processor, x[train_size:], y[train_size:]


def train(x_train, y_train, vocab_processor, x_dev, y_dev, is_baseline, checkpoint_root):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
          with tf.device('/device:GPU:0'):
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
            #     1000, 0.96, staircase=True)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)  # 1e-3
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            # timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", checkpoint_root))#timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            extract_dict(vocab_processor, out_dir)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                # L2 norm constraint (too slow) https://github.com/dennybritz/cnn-text-classification-tf/issues/88
                # sess.run(cnn.output_W.assign(tf.clip_by_norm(cnn.output_W, 1.0)))
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                return loss, accuracy

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_metric_perf = 0.0  # dev_accuracy
            cur_metric_perf = 0.0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    _, cur_metric_perf = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if cur_metric_perf > best_metric_perf:
                    best_metric_perf = cur_metric_perf
                else:
                    continue
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    if len(argv) != 3:
        print("[usage] " + argv[0] + " <1(baseline)/0(test)> <checkpoint_root_dir_name>")
        sys.exit(1)
    is_baseline = bool(int(argv[1]))  # 1 for baseline, 0 for test
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess_nsmc_data(is_baseline)
    #sys.exit(1)
    train(x_train, y_train, vocab_processor, x_dev, y_dev, is_baseline, argv[2])

if __name__ == '__main__':
    tf.app.run()

#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import data_helpers_nsmc
# from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import argparse


# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Data loading params for nsmc only
tf.flags.DEFINE_string("pos_test_data_file", "./data/nsmc/test/toy_nsmc_pos_test.txt", "Data source for the positive test data.")
tf.flags.DEFINE_string("neg_test_data_file", "./data/nsmc/test/toy_nsmc_neg_test.txt", "Data source for the negative test data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def main(is_baseline, checkpoint_dir, pos_test_file, neg_test_file, performance_file):
    FLAGS = tf.flags.FLAGS

    # CHANGE THIS: Load data. Load your own data here
    if is_baseline:
        x_raw, y_test = data_helpers_nsmc.load_nsmc_test_data_and_labels_baseline(pos_test_file, neg_test_file)
    else:
        x_raw, y_test = data_helpers_nsmc.load_nsmc_test_data_and_labels_test(pos_test_file, neg_test_file)
    y_test = np.argmax(y_test, axis=1)

    # Map data into vocabulary
    vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            with tf.device('/device:GPU:0'):
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                batches = data_helpers_nsmc.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

                # Collect the predictions here
                all_predictions = []

                for x_test_batch in batches:
                    batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    out_path = os.path.join(checkpoint_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        perf_file_path = os.path.join(checkpoint_dir, "..", performance_file)#"performance.txt")
        with open(perf_file_path, 'wt') as outf:
            outf.write("Total number of test examples: {}\n".format(len(y_test)))
            outf.write("Accuracy: {:g}\n".format(correct_predictions/float(len(y_test))))

if __name__ == "__main__":
    print("[" + __file__ + "] main invoked.")

    AP = argparse.ArgumentParser(description="args parser")
    AP.add_argument("-is_baseline", action="store", required=True,
                    help="1 if it's for baseline model or 0 for test model")
    AP.add_argument("-checkpoint_dir", action="store", required=True,
                    help="checkpoint path to store the model files in")
    AP.add_argument("-pos_test_file", action="store", required=True,
                    help="data source for the positive test data")
    AP.add_argument("-neg_test_file", action="store", required=True,
                    help="data source for the negative test data")
    AP.add_argument("-performance_file", action="store", required=True,
                    help="performance file name")
    ARGS = AP.parse_args()

    main(bool(ARGS.is_baseline), ARGS.checkpoint_dir,
         ARGS.pos_test_file, ARGS.neg_test_file, ARGS.performance_file)

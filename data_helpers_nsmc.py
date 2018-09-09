import numpy as np
import re
import sys


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_nsmc_data(train_data_file, validate_data_file, for_test_model):
    # Load data from files
    positive_train_examples = []
    negative_train_examples = []
    positive_validate_examples = []
    negative_validate_examples = []

    train_inf = open(train_data_file, "r", encoding='utf-8')
    while True:
        line = train_inf.readline()
        if not line:
            break
        tokens = line.strip().split('\t')
        if len(tokens) != 2:
            print("[ERROR] invalid corpus line found!!!")
            print("\tline: <" + line + ">")
            sys.exit(1)
        text = tokens[1]
        if for_test_model:
            text += (" " + re.sub(r'\s+', '', tokens[1]))
        if tokens[0] == 0:  # negative
            negative_train_examples.append(text)
        else:
            positive_train_examples.append(text)
    train_inf.close()

    validate_inf = open(validate_data_file, "r", encoding='utf-8')
    while True:
        line = validate_inf.readline()
        if not line:
            break
        tokens = line.strip().split('\t')
        text = tokens[1]
        if for_test_model:
            text += ("_$_" + re.sub(r'\s+', '', tokens[1]))
        if toknes[0] == 0:  # negative
            negative_validate_examples.append(text)
        else:
            negative_validate_examples.append(text)
    validate_inf.close()

    return positive_train_examples, negative_train_examples,\
        positive_validate_examples, negative_validate_examples


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

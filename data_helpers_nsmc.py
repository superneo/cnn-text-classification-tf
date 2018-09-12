import numpy as np
import re
import sys


EOS = "EOS"
SPACE = "SPACE"

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


def load_nsmc_train_val_data(pos_train_data_file, pos_validate_data_file,
    neg_train_data_file, neg_validate_data_file, is_test_model):
    # Load data from files
    positive_train_examples = []
    positive_validate_examples = []
    negative_train_examples = []
    negative_validate_examples = []

    max_line_len = 0

    pos_train_inf = open(pos_train_data_file, "r", encoding='utf-8')
    line_cnt = 0
    while True:
        line_cnt += 1
        line = pos_train_inf.readline()
        if not line:
            break
        tokens = line.strip().split('\t')
        if len(tokens) != 2 or tokens[0] != '1' or len(tokens[1].strip()) == 0:
            print("[ERROR] invalid pos train corpus line found!!! (" + str(line_cnt) + ")")
            print("\tline: <" + line + ">")
            sys.exit(1)
        if is_test_model:
            chars = list(tokens[1]) + [EOS] + list(re.sub(r'\s+', '', tokens[1]))
        else:
            chars = list(tokens[1])
        for i, c in enumerate(chars):
            if c == ' ':
                chars[i] = SPACE
        text = ' '.join(chars)
        positive_train_examples.append(text)
        if len(chars) > max_line_len:
            max_line_len = len(chars)
            print("[load_nsmc_train_val_data] max len text: " + str(max_line_len) + " <" + text + ">")
    pos_train_inf.close()

    pos_validate_inf = open(pos_validate_data_file, "r", encoding='utf-8')
    line_cnt = 0
    while True:
        line_cnt += 1
        line = pos_validate_inf.readline()
        if not line:
            break
        tokens = line.strip().split('\t')
        if len(tokens) != 2 or tokens[0] != '1' or len(tokens[1].strip()) == 0:
            print("[ERROR] invalid pos validate corpus line found!!! (" + str(line_cnt) + ")")
            print("\tline: <" + line + ">")
            sys.exit(1)
        if is_test_model:
            chars = list(tokens[1]) + [EOS] + list(re.sub(r'\s+', '', tokens[1]))
        else:
            chars = list(tokens[1])
        for i, c in enumerate(chars):
            if c == ' ':
                chars[i] = SPACE
        text = ' '.join(chars)
        positive_validate_examples.append(text)
        if len(chars) > max_line_len:
            max_line_len = len(chars)
            print("[load_nsmc_train_val_data] max len text: " + str(max_line_len) + " <" + text + ">")
    pos_validate_inf.close()

    neg_train_inf = open(neg_train_data_file, "r", encoding='utf-8')
    line_cnt = 0
    while True:
        line_cnt += 1
        line = neg_train_inf.readline()
        if not line:
            break
        tokens = line.strip().split('\t')
        if len(tokens) != 2 or tokens[0] != '0' or len(tokens[1].strip()) == 0:
            print("[ERROR] invalid neg train corpus line found!!! (" + str(line_cnt) + ")")
            print("\tline: <" + line + ">")
            sys.exit(1)
        if is_test_model:
            chars = list(tokens[1]) + [EOS] + list(re.sub(r'\s+', '', tokens[1]))
        else:
            chars = list(tokens[1])
        for i, c in enumerate(chars):
            if c == ' ':
                chars[i] = SPACE
        text = ' '.join(chars)
        negative_train_examples.append(text)
        if len(chars) > max_line_len:
            max_line_len = len(chars)
            print("[load_nsmc_train_val_data] max len text: " + str(max_line_len) + " <" + text + ">")
    neg_train_inf.close()

    neg_validate_inf = open(neg_validate_data_file, "r", encoding='utf-8')
    line_cnt = 0
    while True:
        line_cnt += 1
        line = neg_validate_inf.readline()
        if not line:
            break
        tokens = line.strip().split('\t')
        if len(tokens) != 2 or tokens[0] != '0' or len(tokens[1].strip()) == 0:
            print("[ERROR] invalid neg validate corpus line found!!! (" + str(line_cnt) + ")")
            print("\tline: <" + line + ">")
            sys.exit(1)
        if is_test_model:
            chars = list(tokens[1]) + [EOS] + list(re.sub(r'\s+', '', tokens[1]))
        else:
            chars = list(tokens[1])
        for i, c in enumerate(chars):
            if c == ' ':
                chars[i] = SPACE
        text = ' '.join(chars)
        negative_validate_examples.append(text)
        if len(chars) > max_line_len:
            max_line_len = len(chars)
            print("[load_nsmc_train_val_data] max len text: " + str(max_line_len) + " <" + text + ">")
    neg_validate_inf.close()

    return positive_train_examples, positive_validate_examples,\
        negative_train_examples, negative_validate_examples, max_line_len


def load_nsmc_test_data_and_labels_test(positive_data_file, negative_data_file):
    print("[load_nsmc_test_data_and_labels_test] invoked for test model")
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip().split('\t')[1] for s in positive_examples]
    for i, line in enumerate(positive_examples):
        syllable_list = list(line)
        for j, c in enumerate(syllable_list):
            if c == ' ':
                syllable_list[j] = SPACE
        positive_examples[i] = ' '.join(syllable_list + [EOS] + list(re.sub(r'\s+', '', line)))

    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip().split('\t')[1] for s in negative_examples]
    for i, line in enumerate(negative_examples):
        syllable_list = list(line)
        for j, c in enumerate(syllable_list):
            if c == ' ':
                syllable_list[j] = SPACE
        negative_examples[i] = ' '.join(syllable_list + [EOS] + list(re.sub(r'\s+', '', line)))

    x_text = positive_examples + negative_examples

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_nsmc_test_data_and_labels_baseline(positive_data_file, negative_data_file):
    print("[load_nsmc_test_data_and_labels_baseline] invoked for baseline model")
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip().split('\t')[1] for s in positive_examples]
    for i, line in enumerate(positive_examples):
        syllable_list = list(line)
        for j, c in enumerate(syllable_list):
            if c == ' ':
                syllable_list[j] = SPACE
        positive_examples[i] = ' '.join(syllable_list)

    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip().split('\t')[1] for s in negative_examples]
    for i, line in enumerate(negative_examples):
        syllable_list = list(line)
        for j, c in enumerate(syllable_list):
            if c == ' ':
                syllable_list[j] = SPACE
        negative_examples[i] = ' '.join(syllable_list)

    x_text = positive_examples + negative_examples

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


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

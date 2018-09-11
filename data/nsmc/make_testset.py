#!/usr/bin/env python3
#-*- coding: utf-8 -*-


import argparse
import random
import sys


def make_test_string(original_string, threshold):
    if not original_string or not threshold:
        print("[ERROR] invalid input argument passed!!!")
        sys.exit(1)

    tokens = original_string.strip().split('\t')
    if len(tokens) != 2:
        print("[ERROR] invalid corpus line found: <" + original_string + ">")
        return None

    text = ""
    for letter in tokens[1]:
        p = random.uniform(0, 1)
        if letter == ' ':
            if p < threshold:
                continue
        else:
            # make threshold for non-spaces much smaller
            # as korean sentence usually has more non-spaces
            if p < threshold / 2.0:
                text += (letter + ' ')
                continue
        text += letter
    tokens[1] = text.strip()  # remove leading & trailing spaces if any

    return tokens[0] + '\t' + tokens[1]


def make_test_corpus_file(input_file, output_file, ratio):
    if not input_file or not output_file or not ratio:
        print("[ERROR] invalid input argument passed!!!")

    inf = open(input_file, "rt")
    outf = open(output_file, "wt")

    while True:
        # ex) 1	볼수록 빠져드는 연출력과 영상미
        line = inf.readline()
        if not line:
            break
        line = make_test_string(line, ratio)
        outf.write(line + '\n')

    outf.close()
    inf.close()


if __name__ == '__main__':
    print("[" + __file__ + "] main invoked.")

    random.seed(925)

    AP = argparse.ArgumentParser(description="args parser")
    AP.add_argument("-input_file", action="store", required=True,
                    help="input corpus file path")
    AP.add_argument("-output_file", action="store", required=True,
                    help="output corpus file path")
    AP.add_argument("-ratio", action="store", default="0.0",
                    help="random space removal ratio")
    ARGS = AP.parse_args()

    make_test_corpus_file(ARGS.input_file, ARGS.output_file, float(ARGS.ratio))

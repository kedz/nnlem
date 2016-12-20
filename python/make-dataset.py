from __future__ import division, print_function
import os
import random
import math

def write_vocab(vocab, path):
    with open(path, "w") as f:
        f.write("<se>\n<sd>\n<ed>\n")
        f.write(u"\n".join(vocab).encode("utf-8"))

def write_dataset(data, path):
    with open(path, "w") as f:
        for word, lemma in data:
            line = u" ".join(word) + u" ||| " + u" ".join(lemma) + u"\n"
            f.write(line.encode("utf-8"))


def preprocess_data(path, output_dir, seed):

    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vocab_path = os.path.join(output_dir, "vocab.txt")
    train_path = os.path.join(output_dir, "lemmas.en.train.txt")
    dev_path = os.path.join(output_dir, "lemmas.en.dev.txt")
    test_path = os.path.join(output_dir, "lemmas.en.test.txt")

    random.seed(seed)

    data = list()
    with open(path, "r") as f:
        for line in f:
             lemma, word = line.decode("utf-8").strip().lower().split()
             data.append((word, lemma))

    data_size = len(data)
    print("Splitting dataset of {:d} datapoints with random seed {:d}".format(
        data_size, seed))

    random.shuffle(data)
    train_start = 0
    train_stop = dev_start = int(math.floor(data_size * .6))
    dev_stop = test_start = int(math.floor(data_size * .9))
    test_stop = data_size

    data_train = data[train_start:train_stop]
    data_dev = data[dev_start:dev_stop]
    data_test = data[test_start:test_stop]
    
    print("")
    print("   Part  |         Size")
    print("=========|=============")
    print("  Train  |   {:10d}".format(len(data_train)))
    print("    Dev  |   {:10d}".format(len(data_dev)))
    print("   Test  |   {:10d}".format(len(data_test)))

    vocab = set()
    for dataset in [data_train, data_dev, data_test]:
        for x, y in data_train:
            for char in x:
                vocab.add(char)
            for char in y:
                vocab.add(char)
    vocab = list(vocab)
    vocab.sort()
    write_vocab(vocab, vocab_path)
    write_dataset(data_train, train_path)
    write_dataset(data_dev, dev_path)
    write_dataset(data_test, test_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare lemma data.')
    parser.add_argument('--path', metavar='P', help='Location of data file.',
                        required=True)
    parser.add_argument('--seed', metavar='S', type=int, help="Random seed.",
                        required=True)
    parser.add_argument('--dest', metavar='P', required=True,
                        help='Directory to write data files')


    args = parser.parse_args() 
    preprocess_data(args.path, args.dest, args.seed)

if __name__ == "__main__":
    main()

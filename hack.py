import sys


if __name__ == '__main__':
    lbl_file = sys.argv[1]
    with open(lbl_file, 'r') as f:
        lbls = f.readlines()[1:]

    with open('data/test.tsv', 'r') as f:
        with open('data/new_test.tsv', 'w') as ff:
            ff.write('index\tquention\tsentence\tlabel\n')
            samples = f.readlines()[1:]
            for sample, lbl in zip(samples, lbls):
                new_sample = sample[:-1] + '\t' + lbl.split('\t')[1]
                ff.write(new_sample)
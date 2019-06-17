import sys


if __name__ == '__main__':
    path1, path2 = sys.argv[1:]
    with open(path1, 'r') as f1:
        with open(path2, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

            cnt = 0
            for l1, l2 in zip(lines1, lines2):
                if l1 != l2:
                    cnt += 1

            print('different pairs: ', cnt)
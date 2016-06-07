#!/usr/bin/python
import sys
import argparse

def tell(args):
    f = open(args.data)
    f_out = open(args.data + '_idx.txt', 'w')

    last_tell = f.tell()

    while True:
        line = f.readline()
        now_tell = f.tell()
        if not line:
            break
        if line.startswith(args.delim):
            if last_tell > 0:
                f_out.write(str(last_tell) + '\n')
            f_out.write(line.strip() + ',' + str(now_tell) + ',')                
        last_tell = now_tell

    f_out.write(str(last_tell) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--delim',
        help='sequence delimiter',
        default='./')
    parser.add_argument(
        '--data',
        help='path to data file',
        default='data')

    args = parser.parse_args()
    tell(args)


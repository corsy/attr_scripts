#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import numpy as np

if sys.platform.startswith('linux'):
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_loss_curve(fname):
    loss_iter = []
    loss = []
    test_iter = []
    test = []
    for line in open(fname):
        if 'Iteration' in line and 'loss' in line:
            txt2 = re.search(ur'Iteration\s([0-9]+)', line)
            txt = re.search(ur'loss\s=\s([0-9\.e-]+)\n', line)
            loss_value = float(txt.groups()[0])
            iter_value = int(txt2.groups()[0])
            loss.append(loss_value)
            loss_iter.append(iter_value)

        if 'Testing net' in line:
            txt = re.search(ur'Iteration\s([0-9]+)', line)
            test_iter.append(int(txt.groups()[0]))
        if 'Test net output' in line and 'loss':
            txt = re.search(ur'=\s*([0-9\.]+)\s*loss\)', line)
            if txt:
                test.append(float(txt.groups()[0]))

    # limit =int(sys.argv[len(sys.argv)-2])
    # while len(loss_iter) and loss_iter[0]<=limit :
    #    del loss_iter[0]
    #    del loss[0]
    # while len(test_iter) and test_iter[0]<=limit :
    #    del test_iter[0]
    #    del test[0]
    plt.plot(loss_iter, loss, color='blue')
    plt.plot(test_iter, test, color='green')


def show_usage():
    print 'python draw_loss.py <log file> <img file>'

if __name__ == '__main__':

    if len(sys.argv) < 2:
        show_usage()

    plt.clf()
    save_loss_curve(sys.argv[1])
    plt.savefig(sys.argv[2])

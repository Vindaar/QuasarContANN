#!/usr/bin/env python

import numpy as np
import fitsio
import theanets

from mlp import load_SDSS_data


def main(args):

    filename = []
    filename.append('../../dr9list.lis')
    
    datasets, size = load_SDSS_data(filename)

    # train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]

    regr = theanets.Experiment(theanets.recurrent.Regressor, layers=(size, 100, 100, size), optimize='sgd', activation='tanh')

    regr.run(datasets[0], datasets[1])

if __name__=="__main__":
    import sys
    main(sys.argv[1:])

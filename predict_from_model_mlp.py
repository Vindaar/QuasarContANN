#!/usr/bin/env python
"""
   Predict from model obtained by mlp.py
"""
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T

#import numpy as np
import fitsio
from SDSSmodules.SDSSfiles import *
from SDSSmodules.SDSSclasses import *


from mlp          import MLP
from matplotlib import pyplot as plt
from handle_data import load_SDSS_predict

def predict_from_mlp(args):
    
    save_file = open('classifier_best_params.mlp')
    layer_params = cPickle.load(save_file)
    mlp_layout   = cPickle.load(save_file)
    save_file.close()

    # create function to predict from classifier
    index = T.lscalar()
    x = T.matrix('x')
    y = T.matrix('y')

    rng = np.random.RandomState(1234)
    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=mlp_layout[0],
        n_hidden=mlp_layout[1],
        n_out=mlp_layout[2],
        layer_params=layer_params,
        ann_layout=mlp_layout
    )

    start = 25000
    end = 25100
    datasets, size, wave_predict = load_SDSS_predict(args, start, end)
    predict_set_x, predict_set_y = datasets[0]
    #predict_x_val = theano.function([x], x)
    #predict_y_val = function([predict_set_y], predict_set_y)
    #print predict_x_val(predict_set_x)
    #print predict_set_x[0].eval()

    predict_from_mlp = theano.function(
        inputs=[index],
        outputs=classifier.y_pred,
        givens={
            x: predict_set_x[index:(index+1)]
        }
    )
    fig, axarr = plt.subplots(1, figsize=(10,8), dpi=100)
    axarr.set_xlim(1000, 1230)

    i = 87
    axarr.plot(wave_predict[i], predict_set_x[i].eval(), 'r-')
    axarr.plot(wave_predict[i], predict_set_y[i].eval(), 'g-')
    print np.shape(predict_from_mlp(i)), predict_set_x[i].eval()
    print predict_from_mlp(i)
    axarr.plot(wave_predict[i], predict_from_mlp(i)[0], 'b-')
    plt.show()
#classifier.y_pred.eval())

if __name__ == '__main__':
    import sys
    predict_from_mlp(sys.argv[1:])

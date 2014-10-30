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

import numpy

import theano
import theano.tensor as T

#import numpy as np
import fitsio
from SDSSmodules.SDSSfiles import *
from SDSSmodules.SDSSclasses import *
from congrid import rebin_1d, get_arrays

from logistic_sgd import LogisticRegression, load_data
from mlp          import MLP
from matplotlib import pyplot as plt

def get_flux_arrays(files):

    #files = open('../../dr9list.lis', 'r').readlines()

    spec = spectrum()

    flux = []
    flux_error = []
    model_sdss = []
    size = []
    wave = []
    nspec = len(files)
    print nspec

    for i in xrange(nspec):
        read_speclya_fitsio(files[i], spec)
        wave_temp = spec.wave / (spec.z + 1.0)
        lya_low = 1041#*(spec.z + 1.0)
        lya_up  = 1185#*(spec.z + 1.0)
        # index = numpy.where((spec.flux_error > 0) &
        #                  (spec.wave > lya_low) &
        #                  (spec.wave < lya_up ) &
        #                  (spec.mask_comb == 0))[0]
        index = numpy.where((wave_temp > lya_low) &
                         (wave_temp < lya_up ))[0]
        size.append(numpy.size(index))
        flux.append(spec.flux[index])
        flux_error.append(spec.flux_error[index])
        model_sdss.append(spec.model_sdss[index])
        wave.append(wave_temp[index])

    max_size = int(max(size))
    for i in xrange(nspec):
        flux[i] = rebin_1d(flux[i], max_size, method='spline')
        wave[i] = rebin_1d(wave[i], max_size, method='spline')
        model_sdss[i] = rebin_1d(model_sdss[i], max_size, method='spline')
        print numpy.shape(flux[i])

    flux = numpy.asarray(flux)
    flux_error = numpy.asarray(flux_error)
    wave = numpy.asarray(wave)
    model_sdss = numpy.asarray(model_sdss)
    min_size = int(min(size))
    flux_ar = numpy.empty((min_size, nspec))    
    model_ar = numpy.empty((min_size, nspec))

    return flux, model_sdss, wave, max_size

def load_SDSS_predict(args):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    print '... loading data'

    # Load the dataset
    # Load spectra as a data set. Train, validate and test
    spectra_f = open(args[0], 'r').readlines()
    predict_set_x, predict_set_y, wave_predict, size_predict = get_flux_arrays(spectra_f[4000:4100])
    
    predict_set = [predict_set_x, predict_set_y]


    #train_set, valid_set, test_set = cPickle.load(f)
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        # basti: need floats, since we want to do regression
        return shared_x, shared_y#T.cast(shared_y, 'int32')

    predict_set_x, predict_set_y = shared_dataset(predict_set)

    rval = [(predict_set_x, predict_set_y)]

    return rval, size_predict, wave_predict


def predict_from_mlp(args, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=3, n_hidden=500):
    
    save_file = open('classifier_params.mlp')
    layer_params = cPickle.load(save_file)
    mlp_layout   = cPickle.load(save_file)
    save_file.close()

    # create function to predict from classifier
    index = T.lscalar()
    x = T.matrix('x')
    y = T.matrix('y')
    datasets, size, wave_predict = load_SDSS_predict(args)

    rng = numpy.random.RandomState(1234)
    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=mlp_layout[0],
        n_hidden=mlp_layout[1],
        n_out=mlp_layout[2],
        layer_params=layer_params
    )

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
    axarr.set_xlim(900, 1500)

    i = 93
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

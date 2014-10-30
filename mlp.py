#!/usr/bin/env python
"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

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

def load_SDSS_data(args, nspec=1000):
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
    train_set_x, train_set_y, wave_train, size_train = get_flux_arrays(spectra_f[0:nspec])
    valid_set_x, valid_set_y, wave_valid, size_valid = get_flux_arrays(spectra_f[nspec + 1 : 2 * nspec + 1])
    test_set_x, test_set_y,   wave_test,  size_test  = get_flux_arrays(spectra_f[nspec * 2 + 2 : 3*nspec + 2])    
    print numpy.shape(train_set_x), numpy.shape(train_set_y)
    train_set = [train_set_x, train_set_y]
    valid_set = [valid_set_x, valid_set_y]
    test_set = [test_set_x, test_set_y]

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

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    if size_train == size_valid == size_test:
        print 'all sizes are the same. Continue.'
    else:
        print 'Error: Sizes of train, valid and test are not the same.'
        import sys
        sys.exit()

    return rval, size_train


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, layer_params=None, ann_layout=None):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        
        :type layer_params: tuple of float
        :param layer_params: weights for all neurons, to load a network
        
        :type ann_layout: tuple of int
        :param ann_layout: tuple of int, giving the number of neurons on layers
        from input to output layer
        e.g.: [10, 5, 5, 15]
        10 input neurons, 2 hidden layers with 5 neurons each, 15 output neurons

        """

        # if layer_params not None, we load a network configuration
        if layer_params is not None:
            W_hid, b_hid = layer_params[0]
            W_logR, b_logR = layer_params[1]
        else:
            W_hid=None
            b_hid=None
            W_logR=None
            b_logR=None

        # check if we use a specfic layout:
        if ann_layout is not None:
            # n_hid_layers is an array of ints, containing the number of neurons on each hidden
            # layer
            n_in = ann_layout[0]
            n_out = ann_layout[-1]
            n_hid_layers = ann_layout[1:-2]
            n_hid = numpy.size(n_hidden)

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        for n_hid in n_
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh,
            W=W_hid,
            b=b_hid
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            W=W_logR,
            b=b_logR
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.prediction_error_sq = (
            self.logRegressionLayer.prediction_error_sq
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        
        # the output prediction is given by 
        self.y_pred = self.logRegressionLayer.y_pred

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.layer_params = [self.hiddenLayer.params, self.logRegressionLayer.params]
        # end-snippet-3


    #def predict(self, x):
        # function, which returns a prediction (a vector) for a given input (a vector)
        # input = flux of spectrum
        # out = prediction

        
        


    # def cost(self, y):
    #     # function to calculate the cost of the MLP Regressor based on the expected value
    #     # y
    #     # get output from LogisticRegression classifier by 
    #     output = self.logRegressionLayer.p_y_given_x
    #     # TODO: find out how to look at shape of output and y
    #     # then adjust such that we calculate:
    #     # take row of y corresponding to training sample 
    #     # subtract output from y
    #     # sum over squares
    #     print y.shape
    #     print output.shape

    #     cost_val  = T.sum((y - output)**2, axis=None)
    #     #cost_val   = error * error
    #     return cost_val


def test_mlp(args, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    #datasets = load_data(dataset)
    datasets, size = load_SDSS_data(args, 50)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    print test_set_x
    print test_set_y
    #print datasets[0]

    # # Load spectra as a data set. Train, validate and test
    # spectra_f = open(args[0], 'r').readlines()
    # train_set_x, train_set_y, wave_train, size_train = get_flux_arrays(spectra_f[0:30])
    # valid_set_x, valid_set_y, wave_valid, size_valid = get_flux_arrays(spectra_f[100:130])
    # test_set_x, test_set_y,   wave_test,  size_test  = get_flux_arrays(spectra_f[5000:5030])

    # if size_train == size_valid == size_test:
    #     print 'yay, we can continue'
    # else:
    #     print 'Error: train, valid and test arrays do not have the same sizes'
    #     import sys
    #     sys.exit()

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the spectra are represented as a matrix
                       # 1 spectrum per row
    y = T.matrix('y', dtype='float64')  # the continuum is presented as a matrix

    print x.type, y.type

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=size,
        n_hidden=n_hidden,
        n_out=size
    )
    mlp_layout = [size, n_hidden, size]

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
         classifier.prediction_error_sq(y)
         + L1_reg * classifier.L1
         + L2_reg * classifier.L2_sqr
    )
    # as error we use: cost = error**2
    # where error: (expected value - output value)
    # 

    #cost = classifier.cost(y)

    #err = y - 
    
    # end-snippet-4

    print 'f', test_set_y[0].shape[1]

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y, threshold=0.01),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y, threshold=0.01),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


    # cPickle classifier (better to pickle weights only!)
    # to use in predicting and plotting script
    save_file = open('classifier_params.mlp', 'wb')
    cPickle.dump(classifier.layer_params, save_file, -1)
    cPickle.dump(mlp_layout, save_file, -1)
    save_file.close()
    

if __name__ == '__main__':
    import sys
    test_mlp(sys.argv[1:])

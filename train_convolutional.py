#!/usr/bin/env python
"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time
import cPickle

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from handle_data import load_SDSS_data
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvNetwork


def evaluate_lenet5(args, learning_rate=0.1, n_epochs=50000,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=5):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    unpickle_data = ''
    if '--unpickle' in args:
        try:
            i = args.index('--unpickle')
            unpickle_data = args[i+1]
        except IndexError:
            import sys
            sys.exit('Error: Provide a file from which to unpickle SDSS data!')

    # Load spectra as a data set. Train, validate and test
    # first check, whether we read data from a pickled file
    if unpickle_data is not '':
        datafile = open(unpickle_data)
        datasets = cPickle.load(datafile)
        size     = cPickle.load(datafile)
    else:
        #datasets = load_data(dataset)
        datasets, size = load_SDSS_data(args, 200, reshape_2D = 0)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    print test_set_x
    print test_set_y

    #TODO: the training data needs to be dust corrected and residual corrected!

    rng = numpy.random.RandomState(23455)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = test_set_x. get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the spectra are represented as a matrix
                       # 1 spectrum per row
    y = T.matrix('y')  # the continuum is presented as a matrix

    print x.type, y.type

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # reshape input to match needed input shape
    layer0_input = x.reshape((batch_size, 1, 44, 44))
    #layer0_input = x.reshape((batch_size, 1, 24, 24))
    print layer0_input.ndim
    #layer0_input = layer0_input.dimshuffle(0, 1, 'x', 2)
    print layer0_input.ndim

    # ann_layout: 
    # 1st entry: number of convolutional layers
    # rest: mlp_layout; 50*3*3 neurons on hidden layer, size = 576 neurons on output
    # 50*3*3 is the number of outputs on the last convolutional layer calculated by
    # nkerns[i-1] * ((image_size - filter_size + 1) / 2)**2 recursively over the 
    # layers. In implementation for further details
    #ann_layout   = (2, 50*3*3, 500, size)
    ann_layout   = (2, nkerns[-1]*8*8, 2000, 60, size)

    classifier = LeNetConvNetwork(
        rng=rng,
        input=layer0_input,
        filter0_shape=(nkerns[0], 1, 5, 5),
        #image0_shape=(batch_size, 1, 24, 24),
        image0_shape=(batch_size, 1, 44, 44),
        poolsize=(2,2),
        nkerns=nkerns,
        ann_layout=ann_layout
    )

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        classifier.prediction_error_sq(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        classifier.prediction_error_sq(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
         classifier.prediction_error_sq(y)
    )
    # as error we use: cost = error**2
    # where error: (expected value - output value)
    # 

    #cost = classifier.cost(y)

    #err = y - 
    
    # end-snippet-4

    print 'f', test_set_y[0].shape[1]#, test_set_y[0].eval()

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.prediction_error_sq(y),#classifier.errors(y, threshold=0.001),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.prediction_error_sq(y),#classifier.errors(y, threshold=0.001),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )


    # create a list of all model parameters to be fit by gradient descent
    params = classifier.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    lr = T.dscalar('lr')
    updates = [
        (param_i, param_i - T.cast(lr, dtype=theano.config.floatX) * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index, theano.Param(lr, default=learning_rate)],
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
    patience = 10000000  # look as this many examples regardless
    patience_increase = 8  # wait this much longer when a new best is
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
    learning_0 = 1
    tau = 0.05

    while (epoch < n_epochs) and (not done_looping):
        try:
            #learning_rate = learning_0 * tau / (tau + epoch) 
            learning_rate = learning_0 / (1 + tau * epoch)
            print 'learning_rate for this epoch:', learning_rate
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    print 'train stuff', minibatch_avg_cost
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
                        best_params = classifier.layer_params

                        # test it on the test set
                        test_losses = [test_model(i) for i
                                       in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    print 'done looping', epoch, iter
                    done_looping = True
                    break
        except KeyboardInterrupt:
            # while network is being trained, check for KeyboardInterrupt
            # so that we can stop the training and save the network\
            # print infos
            print 'Termination initiated...'
            print '...saving networks to unfinishedNetworks folder'
            # cPickle networks (best and current)
            save_file = open('./unfinishedNetworks/classifier_params_unfinished.convmlp', 'wb')
            cPickle.dump(classifier.layer_params, save_file, -1)
            cPickle.dump(ann_layout, save_file, -1)
            save_file.close()

            save_file = open('./unfinishedNetworks/classifier_best_params_unfinished.convmlp', 'wb')
            cPickle.dump(best_params, save_file, -1)
            cPickle.dump(ann_layout, save_file, -1)
            save_file.close()
            
            print '...networks saved'
            
            continue_flag = raw_input(
                'Do you really want to terminate the program?\n'
                'If no, current networks will be saved, but program continues. (Y/n) '
            )
            
            if continue_flag in ['y','Y']:
                import sys
                sys.exit('...terminate')
            else:
                continue


    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    import sys
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


    # cPickle classifier (better to pickle weights only!)
    # to use in predicting and plotting script
    # first cPickle final model
    save_file = open('classifier_params.convmlp', 'wb')
    cPickle.dump(classifier.layer_params, save_file, -1)
    cPickle.dump(ann_layout, save_file, -1)
    save_file.close()
    # now pickle model with best parameters
    save_file = open('classifier_best_params.convmlp', 'wb')
    cPickle.dump(best_params, save_file, -1)
    cPickle.dump(ann_layout, save_file, -1)
    save_file.close()



if __name__ == '__main__':
    import sys
    evaluate_lenet5(sys.argv[1:])
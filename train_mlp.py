#!/usr/bin/env python
"""
script to initialize and train different neural networks
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
from handle_data import load_SDSS_data, convert_flux_to_prediction
from logistic_sgd import LogisticRegression, load_data
from mlp import MLP
from convolutional_mlp import LeNetConvNetwork
from congrid import rebin_1d

# to plot the progress of the network training:
import matplotlib.pyplot as plt



def test_mlp(args, learning_rate=0, L1_reg=0.00, L2_reg=0.0000, n_epochs=250,
             dataset='mnist.pkl.gz', batch_size=5, n_hidden=500):
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

    unpickle_data = ''
    if '--unpickle' in args:
        try:
            i = args.index('--unpickle')
            unpickle_data = args[i+1]
        except IndexError:
            import sys
            sys.exit('Error: Provide a file from which to unpickle SDSS data!')
    if '--train_ann' in args:
        try:
            i = args.index('--train_ann')
            saved_ann = args[i+1]
        except IndexError:
            import sys
            sys.exit('Error: Provide a file from which to read a MLP network to train further!')
        saved_ann = open(saved_ann)
        unsupervised_params = cPickle.load(saved_ann)
        print unsupervised_params
        dbn_layout          = cPickle.load(saved_ann)
        saved_ann.close()
    else:
        unsupervised_params = None
        dbn_layout          = None


    # Load spectra as a data set. Train, validate and test
    # first check, whether we read data from a pickled file
    if unpickle_data is not '':
        datafile = open(unpickle_data)
        datasets = cPickle.load(datafile)
        size     = cPickle.load(datafile)
        size_out = cPickle.load(datafile)
    else:
        #datasets = load_data(dataset)
        datasets, size, size_out = load_SDSS_data(args, 5000, wholespec=True, percentile = 100)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # EXPERIMENT:
    ann_path = '/home/basti/SDSS_indie/Python/MachineLearning/stacked_networks/stacked_best_layer0.convmlp'
    data_new_x = convert_flux_to_prediction(None, size, train_set_x, valid_set_x, test_set_x, ann_path = ann_path, mlp_flag = False)
    ann_path = '/home/basti/SDSS_indie/Python/MachineLearning/stacked_networks/stacked_best_layer1.convmlp'
    data_new_x = convert_flux_to_prediction(None, size, train_set_x, valid_set_x, test_set_x, ann_path = ann_path, mlp_flag = True)
    train_set_x, valid_set_x, test_set_x = data_new_x
    # :EXPERIMENT END
    
    print test_set_x
    print test_set_y

    #TODO: the training data needs to be dust corrected and residual corrected!


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

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    #mlp_layout = [size, n_hidden, size]
    #mlp_layout = [size, 50, 500, 50, size]
    #mlp_layout = [size, 1000, 2000, 60, size]
    mlp_layout = [size, 11000, size_out]
    if dbn_layout == None and unsupervised_params == None:
        classifier = MLP(
            rng=rng,
            input=x,
            activation=T.tanh,
            ann_layout=mlp_layout
        )
    else:
        print 'using previous network weights'
        classifier = MLP(
            rng=rng,
            input=x,
            activation=T.tanh,
            #activation=T.nnet.sigmoid,
            layer_params=unsupervised_params,
            ann_layout=dbn_layout
        )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically

    cost = (
        #classifier.test(y)[-1]
        classifier.prediction_error_sq(y)
        # + L1_reg * classifier.L1
        # + L2_reg * classifier.L2_sqr
    )

    print 'f', test_set_y[0:batch_size].shape[1].eval()#, test_set_y[0].eval()

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
        #outputs=[classifier.test(y)[-1],classifier.prediction_error_sq(y)],
        outputs=classifier.prediction_error_sq(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]
    gparams_pre = [T.grad(classifier.prediction_error_sq(y), param) for param in classifier.params]


    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    # create symbolic varialbe lr (learning rate) to change the learning rate
    # of the network during runtime
    lr = T.dscalar('lr')
    updates = [
        (param, param - T.cast(
                lr,
                dtype=theano.config.floatX
            ) * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    updates_pre = [
        (param, param - T.cast(
                lr,
                dtype=theano.config.floatX
            ) * gparam)
        for param, gparam in zip(classifier.params, gparams_pre)
    ]

    pre_train_model = theano.function(
        inputs=[index, theano.Param(lr, default=learning_rate)],
        outputs=classifier.prediction_error_sq(y),
        updates=updates_pre,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print 'pre training'
    #cost_pre_train = numpy.mean([pre_train_model(i, lr=1) for i in xrange(100)])
    #print 'cost of pre training (prediction_error_sq):', cost_pre_train



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


    test = theano.function(
        inputs=[index],
        outputs=classifier.test(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    # early-stopping parameters
    patience = 2000000  # look as this many examples regardless
    patience_increase = 4  # wait this much longer when a new best is
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
    #tau = 0.075
    tau = 0.05
    epoch_for_learning_rate = 0
    
    train_cost_list = []
    valid_cost_list = []
    valid2_cost_list= []
    test_cost_list  = []

    fig, axarr = plt.subplots(1)
    plt.show(block=False)

    #learning_rate = learning_0 / (1 + tau * epoch)

    while (epoch < n_epochs) and (not done_looping):
        try:
            #learning_rate = learning_0 * tau / (tau + epoch)
            learning_rate = learning_0 / (1 + tau * epoch_for_learning_rate)
            #learning_rate = 0.5
            print 'learning_rate for this epoch:', learning_rate
            epoch = epoch + 1
            epoch_for_learning_rate += 1
            train_avg_cost = []

            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index, learning_rate)
                #print test_model(minibatch_index)
                train_avg_cost.append(minibatch_avg_cost)
                #print 'train stuff', minibatch_avg_cost

                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index
                if (iter + 1) % validation_frequency == 0:
                    # add average train cost to list
                    train_cost_list.append(numpy.mean(train_avg_cost))
                    # compute zero-one loss on validation set
                    print 'train stuff', minibatch_avg_cost
                    #print test(minibatch_index)[-1]
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    # this_validation_loss = numpy.mean(validation_losses[:][1])
                    # this2_validation_loss =numpy.mean(validation_losses[:][0])
                    this_validation_loss = numpy.mean(validation_losses)

                    valid_cost_list.append(this_validation_loss)
                    #valid2_cost_list.append(this2_validation_loss)

                    ########################################
                    ###### Perform some plotting ###########
                    ########################################

                    axarr.clear()
                    axarr.plot(numpy.arange(len(train_cost_list)), train_cost_list)
                    axarr.plot(numpy.arange(len(valid_cost_list)), valid_cost_list)
                    #axarr.plot(numpy.arange(len(valid2_cost_list)), valid2_cost_list)
                    axarr.set_xlabel('Epoch / #')
                    axarr.set_ylabel('Test / Validation cost')
                    plt.draw()
                    plt.savefig('CostPlots/mlp_cost_epochs.png')
                    time.sleep(0.1)
                    plt.pause(0.0001)   

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
                    else:
                        epoch_for_learning_rate += 1
                        learning_rate = learning_0 / (1 + tau * epoch_for_learning_rate)

                if patience <= iter:
                    print 'done looping', epoch, iter
                    done_looping = True
                    break
        except KeyboardInterrupt:
            # while network is being trained, check for KeyboardInterrupt
            # so that we can stop the training and save the network\
            # print infos

            continue_flag = raw_input(
                'Do you really want to terminate the program?\n'
                'If no, current networks will be saved, but program continues. (Y/n) '
            )
            
            if continue_flag in ['y','Y']:
                save_flag = raw_input(
                    'Do you wish to save the current network parameters in ./unfinishedNetworks? (Y/n) '
                )
                if save_flag in ['y','Y']:
                    print '...saving networks to unfinishedNetworks folder'
                    # cPickle networks (best and current)
                    save_file = open('./unfinishedNetworks/classifier_params_unfinished.mlp', 'wb')
                    cPickle.dump(classifier.layer_params, save_file, -1)
                    cPickle.dump(mlp_layout, save_file, -1)
                    save_file.close()
                    save_file = open('./unfinishedNetworks/classifier_best_params_unfinished.mlp', 'wb')
                    cPickle.dump(best_params, save_file, -1)
                    cPickle.dump(mlp_layout, save_file, -1)
                    save_file.close()
                    print '...networks saved'
                import sys
                sys.exit('...terminate')
            else:
                print '...saving networks to unfinishedNetworks folder'
                # cPickle networks (best and current)
                save_file = open('./unfinishedNetworks/classifier_params_unfinished.mlp', 'wb')
                cPickle.dump(classifier.layer_params, save_file, -1)
                cPickle.dump(mlp_layout, save_file, -1)
                save_file.close()
                save_file = open('./unfinishedNetworks/classifier_best_params_unfinished.mlp', 'wb')
                cPickle.dump(best_params, save_file, -1)
                cPickle.dump(mlp_layout, save_file, -1)
                save_file.close()
                print '...networks saved'
                learn_flag = raw_input(
                    'Do you wish to change the learning rate? (Y/n) '
                )
                if learn_flag in ['y', 'Y']:
                    learning_0 = float(raw_input('Please give a new learning rate: '))
                    epoch_for_learning_rate = 0
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
    save_file = open('classifier_params.mlp', 'wb')
    cPickle.dump(classifier.layer_params, save_file, -1)
    cPickle.dump(mlp_layout, save_file, -1)
    save_file.close()
    # now pickle model with best parameters
    save_file = open('classifier_best_params.mlp', 'wb')
    cPickle.dump(best_params, save_file, -1)
    cPickle.dump(mlp_layout, save_file, -1)
    save_file.close()

if __name__ == '__main__':
    import sys
    test_mlp(sys.argv[1:])

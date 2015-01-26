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

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
try:
    from mlp import MLP
except ImportError:
    print 'MLP already imported!'


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), W=None, b=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))

        # check if we load a network with specific weights:
        if W is None:
            # initialize weights with random weights

            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.W = W

        if b is None:
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class LeNetConvNetwork(object):
    """
    Implementation of a LeNet Convolutional MLP network, based on several
    convolutional pooling layers.
    """
    def __init__(self, rng, input, filter0_shape, image0_shape, poolsize, nkerns,  ann_layout, activation = T.tanh, layer_params=None):
        """
        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image0_shape

        :type filter0_shape: tuple or list of length 4
        :param filter0_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image0_shape: tuple or list of length 4
        :param image0_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        
        :type nkerns: tuple
        :param nkerns: number of kernels on each layer

        :type ann_layout: tuple 
        :param ann_layout: layout of the network. 1st entry: number of convolutional
         pooling layers, rest describes MLP at end (# of neurons)
        
        :type layer_params: list numpy arrays
        :param layer_params: if given, load a network with given layout and weights
        """

        # TODO: check if input is of shape (batch size, 1, a, a)
        # where a is any multiple of 2, i.e. 2D input
        # else reshape it

        # check the layout of the network:
        # n_on_hid_layers is an array of ints, containing the number of neurons on each hidden
        # layer
        # n_hid_layers is the number of hidden layers we have
        # mlp_layout is a tuple describing the layout of the mlp network, which sits
        # behind the convolutional pooling layers
        n_conv_layers = ann_layout[0]
        mlp_layout    = ann_layout[1:]
        self.convolutionalLayer = []

        # if layer_params not None, we load a network configuration
        if layer_params is not None:
            # first assert that number of given weights and biases is # of conv layers + layers in mlp:
            # assert np.size(layer_params)
            # first element of layer_params should be the parameters for all convolutional layers
            # elements afterwards contain weights for MLP, since layer_params is created by:
            # layer_params = [convlayers layer_params, mlp layer_params]
            conv_layer_params = layer_params[0]
            mlp_layer_params  = layer_params[1:]
            # now extract individual parts of params from conv_layer_params
            W_conv = []
            b_conv = []
            try:
                for layer in conv_layer_params:
                    print layer, layer[0], layer[1]
                    W_conv_temp, b_conv_temp = layer
                    W_conv.append(W_conv_temp)
                    b_conv.append(b_conv_temp)
            except TypeError:
                for i in xrange(len(conv_layer_params)/2):
                    W_conv_temp = conv_layer_params[2*i]
                    b_conv_temp = conv_layer_params[2*i+1]
                    W_conv.append(W_conv_temp)
                    b_conv.append(b_conv_temp)
            W_conv  = np.asarray(W_conv)
            b_conv  = np.asarray(b_conv)
        else:
            W_conv=[None for _ in xrange(n_conv_layers)]
            b_conv=[None for _ in xrange(n_conv_layers)]
            mlp_layer_params=None

        # TODO: figure out a way to assert those things!
            
        # try:
        #     assert np.size(np.shape(input)) == 4
        #     assert input[-1] == input[-2]
        # except AssertionError:
        #     if input[-1] % 2 != 0:
        #         print input[-1].eval()
        #         import sys
        #         sys.exit('FatalError: Input to network needs to be a multiple of 2!')
        #     else:
        #         input.reshape((input[0], 1, np.sqrt(input[-1]), np.sqrt(input[-1])))

        # print input.eval()
        # assert np.size(input.shape[0]) == 4
        # image_shape = np.shape(input)

        # how does this assertion make sense?
        print np.size(W_conv), n_conv_layers
        assert np.size(W_conv) == n_conv_layers

        # first construct convolutional pooling layers:
        # number of them given by ann_layout[0]:
        for i in xrange(n_conv_layers):
            if i == 0:
                image_shape  = image0_shape
                filter_shape = filter0_shape
                layer_input  = input
                image_size   = image0_shape[-1]
                filter_size  = filter0_shape[-1]
            else:
                image_size   = (image_size - filter_size + 1) / 2
                image_shape  = (image0_shape[0], nkerns[i-1], image_size, image_size)
                filter_shape = (nkerns[i], nkerns[i-1], filter0_shape[2], filter0_shape[3])
                layer_input  = self.convolutionalLayer[i-1].output
            self.convolutionalLayer.append(
                LeNetConvPoolLayer(
                    rng,
                    input=layer_input,
                    image_shape=image_shape,
                    filter_shape=filter_shape,
                    poolsize=poolsize,
                    W=W_conv[i],
                    b=b_conv[i]
                )
            )
        conv_final_output_size = (image_size - filter_size + 1) / 2

        # Now we need to create the MLP, which sits behind the convolutional layers
        # to do that, first flatten output of last conv layer

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        mlp_input = self.convolutionalLayer[-1].output.flatten(2)

        # n_in of first layer of MLP needs to be:
        # nkerns[-1] * conv_final_output_size ** 2
        assert mlp_layout[0] == nkerns[-1] * conv_final_output_size ** 2

        from mlp import MLP
        print 'mlp_layout in convolutional_mlp.py', mlp_layout
        print 'mlp_layer_params in convolutional_mlp.py', mlp_layer_params

        self.MLP = MLP(
            rng=rng,
            input=mlp_input,
            ann_layout=mlp_layout,
            activation = activation,
            layer_params=mlp_layer_params
        )

        # as error we use: cost = error**2
        # where error: (expected value - output value)
        # taken from MLP
        self.prediction_error_sq = self.MLP.prediction_error_sq
        self.test = self.MLP.test

        # the output prediction is given by the MLPs prediction
        self.y_pred = self.MLP.y_pred

        # create a list of all model parameters to be fit by gradient descent
        #params = layer3.params + layer2.params + layer1.params + layer0.params
        #self.layer_params = 

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        # create the self.params and layer_params from all conv layers
        convLayerParams      = []
        convLayerParamsTuple = []
        for layer in self.convolutionalLayer:
            convLayerParams += layer.params
            convLayerParamsTuple.append(layer.params)

        self.params = convLayerParams + self.MLP.params
        self.layer_params = [convLayerParamsTuple, self.MLP.layer_params]

        print 'params'
        print self.params, self.layer_params



def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)

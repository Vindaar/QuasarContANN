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


import numpy

import theano
import theano.tensor as T
from handle_data import load_SDSS_data
from logistic_sgd import LogisticRegression, load_data

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

    def __init__(self, rng, input, ann_layout, activation=T.tanh, layer_params=None):
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
        # check if we use a specfic layout:
        if ann_layout is not None:
            # n_on_hid_layers is an array of ints, containing the number of neurons on each hidden
            # layer
            # n_hid_layers is the number of hidden layers we have
            n_in = ann_layout[0]
            n_out = ann_layout[-1]
            n_on_hid_layer = ann_layout[:-1]
            n_hid_layers = numpy.size(n_on_hid_layer)
            self.hiddenLayer = []
        else:
            n_on_hid_layer = list(n_hidden)
            n_hid_layers = 2

        # if layer_params not None, we load a network configuration
        if layer_params is not None:
            # if we load a network, the number of elements in layer_params - 1 is the number of
            # hidden layers in that network
            if len(layer_params) == 1:
                layer_params = layer_params[0]
            W_hid = []
            b_hid = []
            try:
                for layer in layer_params:
                    W_hid_temp, b_hid_temp = layer
                    W_hid.append(W_hid_temp)
                    b_hid.append(b_hid_temp)
            except TypeError:
                for i in xrange(len(layer_params)/2):
                    W_hid_temp = layer_params[2*i]
                    b_hid_temp = layer_params[2*i+1]
                    W_hid.append(W_hid_temp)
                    b_hid.append(b_hid_temp)
            W_logR = W_hid[-1]
            b_logR = b_hid[-1]
            W_hid  = numpy.asarray(W_hid[:-1])
            b_hid  = numpy.asarray(b_hid[:-1])
        else:
            W_hid=[None for _ in xrange(n_hid_layers)]
            b_hid=[None for _ in xrange(n_hid_layers)]
            W_logR=None
            b_logR=None

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        # now several layers. 1st hidden:
        # self.hiddenLayerIn = HiddenLayer(
        #     rng=rng,
        #     input=input,
        #     n_in=n_in,
        #     n_out=n_on_hid_layers[0],
        #     activation=T.tanh,
        #     W=W_hid[0],
        #     b=b_hid[0]
        # )

        # create several more hidden layers:
        print W_hid, b_hid
        print n_hid_layers

        assert n_hid_layers

        for i in xrange(n_hid_layers - 1):
            # for i == 0 n_on_hid_layer contains the number of input neurons
            n_input   = n_on_hid_layer[i]
            n_output  = n_on_hid_layer[i+1]
            print 'neurons on layer', i
            print n_input, n_output
            if i == 0:
                layer_input = input
            else:
                layer_input = self.hiddenLayer[i-1].output
            self.hiddenLayer.append(
                HiddenLayer(
                rng=rng,
                input=layer_input,
                n_in=n_input,
                n_out=n_output,
                activation=activation,
                W=W_hid[i],
                b=b_hid[i]
            ))

        print 'neurons on layer out'
        print n_on_hid_layer[-1], n_out
        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer[-1].output,
            n_in=n_on_hid_layer[-1],
            n_out=n_out,
            W=W_logR,
            b=b_logR
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        # To calculate L1 and L2, we need the absolute value of the sum of
        # all weights in all layers:
        W_sum  = 0
        W2_sum = 0
        for layer in self.hiddenLayer:
            W_sum  += abs(layer.W).sum()
            W2_sum += (layer.W ** 2).sum()

        self.L1 = (
            W_sum
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            W2_sum
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.prediction_error_sq = (
            self.logRegressionLayer.prediction_error_sq
        )
        self.test = self.logRegressionLayer.test

        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        
        # the output prediction is given by 
        self.y_pred = self.logRegressionLayer.y_pred

        # only of interest if actually no MLP, but DBN
        # gives output of last RBM in DBN
        # self.y_pred_rbms = self.hiddenLayer[-1].output

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        # create the self.params and layer_params from all hidden layers
        hiddenLayerParams = []
        hiddenLayerParamsTuple = []
        for layer in self.hiddenLayer:
            hiddenLayerParams += layer.params
            hiddenLayerParamsTuple.append(layer.params)
        hiddenLayerParamsTuple.append(self.logRegressionLayer.params)

        self.params = hiddenLayerParams  + self.logRegressionLayer.params
        self.layer_params = hiddenLayerParamsTuple
        print 'params'
        print self.params, self.layer_params
        # end-snippet-3


    #def predict(self, x):
        # function, which returns a prediction (a vector) for a given input (a vector)
        # input = flux of spectrum
        # out = prediction


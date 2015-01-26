import numpy as np
import fitsio
from SDSSmodules.SDSSfiles import *
from SDSSmodules.SDSSclasses import *

from congrid import rebin_1d, get_arrays

import theano
import theano.tensor as T

import cPickle

from convolutional_mlp import LeNetConvNetwork
#from mlp import MLP

def get_flux_arrays(files, start, nspec, reshape_2D=0, wholespec = True, percentile = 100, predict = False):

    #files = open('../../dr9list.lis', 'r').readlines()
    filetype = check_filetype(files[0])

    spec = spectrum()

    flux = []
    flux_error = []

    model_sdss = []
    size = []
    wave = []
    bad_pca = open('bad_pca.lis', 'a')

    # We use a while loop instead of for, because we want a specific number of spectra
    # if something is wrong with a spectrum, we skip it and would end up with less 
    # spectra than we want, if we were to use a for loop
    i = 0
    read_specs = 0
    while read_specs < nspec:
        # we assert that the user who calls the function takes care not to try
        # to read more files than in the list file
        assert i < np.size(files)

        if read_specs % 500 == 0:
            print read_specs, 'spectra read'
        
        if filetype == 1:
            read_spSpec_fitsio(files[start+i], spec, None)
        if filetype == 2:
            read_spec_fitsio(files[start+i], spec, None)
        if filetype == 3:
            read_speclya_fitsio(files[start+i], spec, None)
        if filetype == 4:
            read_mockspec_fitsio(files[start+i], spec)
        wave_temp = spec.wave / (spec.z + 1.0)
        if wholespec == True:
            lya_low = 1030
            lya_up  = 1585
        else:
            lya_low = 1041#*(spec.z + 1.0)
            lya_up  = 1175#*(spec.z + 1.0)
        # index = np.where((spec.flux_error > 0) &
        #                  (spec.wave > lya_low) &
        #                  (spec.wave < lya_up ) &
        #                  (spec.mask_comb == 0))[0]

        index = np.where((wave_temp > lya_low) &
                            (wave_temp < lya_up ) &
                            (np.isfinite(spec.flux) == True))[0]

        # before we work on the spectra, we need to reduce the magnitude
        # of their values (otherwise the non linear functions in the hidden
        # layer will be too easily saturated)
        # done by simple division of 90th percentile value in spectrum
        if index != []:
            if percentile is not None:
                regulator = np.percentile(spec.flux[index], percentile)
            else:
                regulator = 1
        else:
            regulator = 1
        #regulator = 1
        # perform mahalanobis scaling:
        flux_temp = spec.flux[index]
        mean = np.mean(flux_temp)
        std  = np.std(flux_temp)
         # doesn't make sense! Can't revert prediction from network this way
        # model_temp = spec.model_sdss[index]
        # mean_model = np.mean(model_temp)
        # std_model  = np.std(model_temp)


        if filetype in [3, 4]:
            if spec.PCA_qual != 1:
                bad_pca.write(spec.filename+'\n')
                print 'bad PCA:', spec.filename
                continue
            else:
                fluxtemp2 = (flux_temp - mean) / std
                ones = np.empty(10)
                ones.fill(1.0)
                if predict == False and np.size(fluxtemp2) == 0:
                    i += 1
                    continue
                #model_sdss.append((spec.model_sdss[index] - mean) / std)# / regulator)
                model_sdss.append(spec.model_sdss[index] / regulator)

        #fluxtemp2 = np.convolve(fluxtemp2, ones, mode='full')
        if np.size(index) > 0:
            fluxtemp2 = np.convolve(spec.flux[index], ones, mode='full')
        fluxtemp2 = fluxtemp2 / (np.size(ones) * regulator)
        flux.append(fluxtemp2)

        #flux.append((flux_temp - mean) / std)
        #flux.append(spec.flux[index] / regulator)
        flux_error.append(spec.flux_error[index])
        size.append(np.size(index))
        wave.append(wave_temp[index])
        i += 1
        read_specs += 1

    max_size = int(max(size))
    print max_size
    if wholespec == True:
        max_size       = 1936 # 44**2
        model_max_size = 1936
    else:
        max_size = 576
    for i in xrange(nspec):
        flux[i] = rebin_1d(flux[i], max_size, method='spline')
        wave[i] = rebin_1d(wave[i], max_size, method='spline')
        if filetype in [3, 4]:
            model_sdss[i] = rebin_1d(model_sdss[i], model_max_size, method='spline')

    flux = np.asarray(flux)
    print 'nan?', np.mean(np.sum(flux, axis=1), axis=0)
    print np.mean(flux, axis=1)[100:300]

    flux_error = np.asarray(flux_error)
    wave = np.asarray(wave)
    if filetype in [3, 4]:
        model_sdss = np.asarray(model_sdss)
    min_size = int(min(size))
    flux_ar = np.empty((min_size, nspec))    
    model_ar = np.empty((min_size, nspec))

    bad_pca.close()

    if reshape_2D == 1:
        np.reshape(flux, (nspec, 24, 24))
        #np.reshape(flux_error, (nspec, 24, 24))
        if filetype in [3, 4]:
            np.reshape(model_sdss, (nspec, 24, 24))

    # variable which counts last file that was read, so that next dataset
    # can start reading from that file
    end = start + i + 1

    if filetype in [3, 4]:
        return end, flux, model_sdss, wave, max_size, model_max_size
    else:
        return end, flux, wave, max_size, model_max_size

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
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

def load_SDSS_data(args, nspec=1000, reshape_2D=0, wholespec = True, percentile = 100):
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

    # variables from where to start reading of each dataset: 
    # start_0, start_1, start_2
    # define start_1 and start_2 for transparency reasons
    start_0 = 0
    start_1 = 0
    start_2 = 0

    start_1, train_set_x, train_set_y, wave_train, size_train, size_out = get_flux_arrays(
        spectra_f,
        start_0,
        nspec,
        reshape_2D = reshape_2D,
        wholespec = wholespec,
        percentile = percentile
    )
    if nspec > 5000:
        nspec_valid = 5000
    else:
        nspec_valid = nspec

    # changed from same number to half of nspec!!!
    start_2, valid_set_x, valid_set_y, wave_valid, size_valid, size_valid_out = get_flux_arrays(
        spectra_f,
        start_1,
        nspec_valid,
        reshape_2D = reshape_2D,
        wholespec = wholespec,
        percentile = percentile
    )
    end, test_set_x, test_set_y,   wave_test,  size_test, size_test_out = get_flux_arrays(
        spectra_f,
        start_2,
        nspec_valid,
        reshape_2D = reshape_2D,
        wholespec = wholespec,
        percentile = percentile
    )

    train_set = [train_set_x, train_set_y]
    valid_set = [valid_set_x, valid_set_y]
    test_set = [test_set_x, test_set_y]

    #input is an np.ndarray of 2 dimensions (a matrix)
    #which row's correspond to an example. target is a
    #np.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    print test_set_x
    print test_set_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    # cPickle the data, so we don't need to read it all again
    path_data = './data/datasets_' + str(nspec) + '_' + str(size_train) + '_' + str(size_out) + '_' + str(percentile) + '_' + str(check_filetype(spectra_f[0])) + '.dat'
    datafile  = open(path_data, 'wb')
    cPickle.dump(rval, datafile, -1)
    cPickle.dump(size_train, datafile, -1)
    cPickle.dump(size_out, datafile, -1)
    datafile.close()
    
    if size_train == size_valid == size_test:
        print 'all sizes are the same. Continue.'
    else:
        print 'Error: Sizes of train, valid and test are not the same.'
        import sys
        sys.exit()

    return rval, size_train, size_out


def load_SDSS_predict(args, start, end, filelist=None, reshape_2D = 0, wholespec = True, percentile = 100):
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
    if filelist is None:
        spectra_f = open(args[0], 'r').readlines()
    else:
        spectra_f = filelist

    # check the filetype. only if speclya files, we will get the y 'predict' set
    filetype = check_filetype(filelist[0])
    nspec = end - start
    if filetype in [3, 4]:
        end, predict_set_x, predict_set_y, wave_predict, size_predict, size_out = get_flux_arrays(
            spectra_f,
            start, 
            nspec,
            reshape_2D = reshape_2D,
            wholespec = wholespec,
            percentile = percentile,
            predict = True
        )
        predict_set = [predict_set_x, predict_set_y]
        predict_set_x, predict_set_y = shared_dataset(predict_set)
        rval = [(predict_set_x, predict_set_y)]
    else:
        end, predict_set_x, wave_predict, size_predict, size_out = get_flux_arrays(
            spectra_f,
            start, 
            nspec,
            reshape_2D = reshape_2D,
            predict = True
        )
        predict_set_x = theano.shared(np.asarray(predict_set_x,
                                                    dtype=theano.config.floatX),
                                      borrow=True)
        rval = predict_set_x

    # cPickle the data, so we don't need to read it all again
    path_data = './data_predict/datasets_' + str(start) + '_' + str(end) + '_' + str(size_predict) + '.dat'
    datafile  = open(path_data, 'wb')
    cPickle.dump(rval, datafile, -1)
    cPickle.dump(wave_predict, datafile, -1)
    cPickle.dump(size_predict, datafile, -1)
    cPickle.dump(size_out, datafile, -1)
    datafile.close()

    return rval, size_predict, size_out, wave_predict


def convert_flux_to_prediction(args, size, data0_x, data1_x = None, data2_x = None, ann_path = None, mlp_flag = False):
    # mlp_flag: type bool
    # if True, we use a MLP network given by the path to change the inputs
    # if False, we use a LeNet

    # ann_path: type string
    # can be used to give a path to a file. If given, args is irrelevant

    # size: type int
    # not used currently

    #### EXPERIMENT!!!
    # Exchange spec.flux with prediction of best neural network so far and
    # use that dataset to feed a normal MLP
    if args is not None and '--convert_by_ann' in args:
        try:
            i = args.index('--convert_by_ann')
            saved_ann_str = args[i+1]
        except IndexError:
            import sys
            sys.exit('If you provide --convert_by_ann option, also need to supply a file')
    else:
        saved_ann_str = 'classifier_best_params.mlp'
    # in case we simply wish to give a path without messing with an args keyword, we can
    # use the ann_path variable
    print 'args none?', args is None
    if args is None and ann_path is not None:
        saved_ann_str = ann_path
    x = T.matrix('x')
    y = T.matrix('y')
    index = T.lscalar()
    
    if data1_x is not None and data2_x is not None:
        data_x = [data0_x, data1_x, data2_x]
    else:
        data_x = [data0_x]

    saved_ann = open(saved_ann_str)
    layer_params = cPickle.load(saved_ann)
    ann_layout   = cPickle.load(saved_ann)
    saved_ann.close()
    batch_size = 1
    rng = np.random.RandomState(1234)
    if mlp_flag == False:
        nkerns = [20, 50]
        #layer0_input = x.reshape((batch_size, 1, 24, 24))
        layer0_input = x.reshape((batch_size, 1, 44, 44))
        classifier = LeNetConvNetwork(
            rng=rng,
            input=layer0_input,
            filter0_shape=(nkerns[0], 1, 5, 5),
            #image0_shape=(batch_size, 1, 24, 24),
            image0_shape=(batch_size, 1, 44, 44),
            poolsize=(2,2),
            nkerns=nkerns,
            ann_layout=ann_layout,
            activation=T.tanh,
            layer_params = layer_params
        )
    else:
        from mlp import MLP
        classifier = MLP(
            rng=rng,
            input=x,
            activation=T.tanh,
            ann_layout=ann_layout
        )

    data_new_x = []
    print ''
    print data_x[0].shape[0].eval()
    print data_x[0].shape[1].eval()
    print ann_layout
    for l, data in enumerate(data_x):
        predict_from_mlp = theano.function(
            inputs=[index],
            outputs=[
                classifier.y_pred
            ],
            givens={
                x: data[index:(index+1)]
            }
        )
        data_temp = []
        for i in xrange(data_x[l].shape[0].eval()):
            if i % 1000 == 0:
                print i, 'flux arrays replaced with predictions'
            temp = predict_from_mlp(i)[0][0]
            temp = rebin_1d(temp, 1936, method='spline')
            data_temp.append(temp)

        data_new = theano.shared(np.asarray(data_temp,
                                               dtype=theano.config.floatX),
                                 borrow = True)
        data_new_x.append(data_new)
        
    return data_new_x


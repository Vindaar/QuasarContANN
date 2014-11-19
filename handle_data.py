import numpy
import fitsio
from SDSSmodules.SDSSfiles import *
from SDSSmodules.SDSSclasses import *
from congrid import rebin_1d, get_arrays

import theano
import theano.tensor as T

import cPickle

def get_flux_arrays(files, reshape_2D=0):

    #files = open('../../dr9list.lis', 'r').readlines()
    filetype = check_filetype(files[0])

    spec = spectrum()

    flux = []
    flux_error = []

    model_sdss = []
    size = []
    wave = []
    nspec = len(files)

    for i in xrange(nspec):
        if i % 500 == 0:
            print i, 'spectra read'
        
        if filetype == 1:
            read_spSpec_fitsio(files[i], spec, None)
        if filetype == 2:
            read_spec_fitsio(files[i], spec, None)
        if filetype == 3:
            read_speclya_fitsio(files[i], spec, None)
        wave_temp = spec.wave / (spec.z + 1.0)
        #lya_low = 1041#*(spec.z + 1.0)
        #lya_up  = 1185#*(spec.z + 1.0)
        lya_low  = 1030
        lya_up   = 1585
        # index = numpy.where((spec.flux_error > 0) &
        #                  (spec.wave > lya_low) &
        #                  (spec.wave < lya_up ) &
        #                  (spec.mask_comb == 0))[0]
        index = numpy.where((wave_temp > lya_low) &
                            (wave_temp < lya_up ) &
                            (np.isfinite(spec.flux) == True))[0]
        size.append(numpy.size(index))
        # before we work on the spectra, we need to reduce the magnitude
        # of their values (otherwise the non linear functions in the hidden
        # layer will be too easily saturated)
        # done by simple division of 90th percentile value in spectrum
        if index != []:
            regulator = np.percentile(spec.flux[index], 100)
        else:
            regulator = 1
        #regulator = 1
        flux.append(spec.flux[index] / regulator)
        flux_error.append(spec.flux_error[index])
        if filetype == 3:
            model_sdss.append(spec.model_sdss[index] / regulator)
        wave.append(wave_temp[index])

    max_size = int(max(size))
    print max_size
    #max_size = 576
    max_size = 1936 # 44**2
    for i in xrange(nspec):
        flux[i] = rebin_1d(flux[i], max_size, method='spline')
        wave[i] = rebin_1d(wave[i], max_size, method='spline')
        if filetype == 3:
            model_sdss[i] = rebin_1d(model_sdss[i], max_size, method='spline')

    flux = numpy.asarray(flux)
    flux_error = numpy.asarray(flux_error)
    wave = numpy.asarray(wave)
    if filetype == 3:
        model_sdss = numpy.asarray(model_sdss)
    min_size = int(min(size))
    flux_ar = numpy.empty((min_size, nspec))    
    model_ar = numpy.empty((min_size, nspec))

    if reshape_2D == 1:
        np.reshape(flux, (nspec, 24, 24))
        #np.reshape(flux_error, (nspec, 24, 24))
        if filetype == 3:
            np.reshape(model_sdss, (nspec, 24, 24))

    if filetype == 3:
        return flux, model_sdss, wave, max_size
    else:
        return flux, wave, max_size

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

def load_SDSS_data(args, nspec=1000, reshape_2D=0):
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
    train_set_x, train_set_y, wave_train, size_train = get_flux_arrays(spectra_f[0:nspec], reshape_2D = reshape_2D)
    if nspec > 5000:
        nspec_valid = 5000
    else:
        nspec_valid = nspec

    valid_set_x, valid_set_y, wave_valid, size_valid = get_flux_arrays(spectra_f[nspec + 1 : nspec + nspec_valid + 1], reshape_2D = reshape_2D)
    test_set_x, test_set_y,   wave_test,  size_test  = get_flux_arrays(spectra_f[nspec + nspec_valid + 2 : nspec + 2*nspec_valid + 2], reshape_2D = reshape_2D)    

    train_set = [train_set_x, train_set_y]
    valid_set = [valid_set_x, valid_set_y]
    test_set = [test_set_x, test_set_y]

    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #which row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    # cPickle the data, so we don't need to read it all again
    path_data = './data/datasets_' + str(nspec) + '_' + str(size_train) + '.dat'
    datafile  = open(path_data, 'wb')
    cPickle.dump(rval, datafile, -1)
    cPickle.dump(size_train, datafile, -1)
    datafile.close()
    
    if size_train == size_valid == size_test:
        print 'all sizes are the same. Continue.'
    else:
        print 'Error: Sizes of train, valid and test are not the same.'
        import sys
        sys.exit()

    return rval, size_train


def load_SDSS_predict(args, start, end, filelist=None, reshape_2D = 0):
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
    if filetype == 3:
        predict_set_x, predict_set_y, wave_predict, size_predict = get_flux_arrays(spectra_f[start:end], reshape_2D = reshape_2D)
        predict_set = [predict_set_x, predict_set_y]
        predict_set_x, predict_set_y = shared_dataset(predict_set)
        rval = [(predict_set_x, predict_set_y)]
    else:
        predict_set_x, wave_predict, size_predict = get_flux_arrays(spectra_f[start:end], reshape_2D = reshape_2D)
        predict_set_x = theano.shared(numpy.asarray(predict_set_x,
                                                    dtype=theano.config.floatX),
                                      borrow=True)
        rval = predict_set_x


    return rval, size_predict, wave_predict

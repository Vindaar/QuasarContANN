#!/usr/bin/env python

# This program plots the spectrum of a FITS file, the
# PCA model and the prediction of the ANN

from copy import deepcopy
import sys
from SDSSmodules.SDSSclasses import spectrum, program_settings
from SDSSmodules.SDSSfitting import fit_powerlaw_individual
from SDSSmodules.SDSSfiles   import *
from SDSSmodules.SDSSutilities import calc_zem_index
import numpy as np
import fitsio
from pylab import *
from matplotlib.backend_bases import Event
from matplotlib import pyplot as plt

import theano
import theano.tensor as T
import cPickle
from mlp          import MLP
from handle_data import load_SDSS_predict
from convolutional_mlp import LeNetConvNetwork
from congrid import rebin_1d

# class which contains all necessary functions to work on the matplotlib graph
class WorkOnSpectrum:
    def __init__(self, filelist, figure, ax, dustmap, settings, resid_corr, classifier, x):
        self.filelist  = filelist
        self.filetype  = check_filetype(filelist[0])
        self.fig       = figure
        self.axarr     = ax
        self.dustmap   = dustmap
        self.nspec     = len(filelist)
        self.settings  = settings
        self.resid_corr= resid_corr
        # we read the starting index from the command line on initialization 
        self.start_iter= int(raw_input('Give the index of the first element to be plotted from the input file: '))
        self.i         = self.start_iter
        # and read the first 100 elements starting from there
        datasets, size, self.wave_predict = load_SDSS_predict(None, self.i, self.i+100, filelist=self.filelist)
        if self.filetype == 3:
            self.predict_set_x, self.predict_set_y = datasets[0]
        else:
            # if not speclya files, only set_x is returned
            self.predict_set_x = datasets

        self.classifier= classifier
        self.index = T.lscalar()
        self.x     = x
        self.y     = T.matrix('y')
        print np.size(self.predict_set_x[0].eval())
        if self.filetype == 3:
            self.predict_from_mlp = theano.function(
                inputs=[self.index],
                outputs=[
                    self.classifier.y_pred,
                    #self.classifier.hiddenLayer[0].output,
                    self.classifier.prediction_error_sq(self.y)
                ],#hiddenLayer[0].output,
                givens={
                    self.x: self.predict_set_x[self.index:(self.index+1)],
                    self.y: self.predict_set_y[self.index:(self.index+1)]
                }
            )
        else:
            self.predict_from_mlp = theano.function(
                inputs=[self.index],
                outputs=[
                    self.classifier.y_pred
                    #self.classifier.hiddenLayer[0].output,
                ],#hiddenLayer[0].output,
                givens={
                    self.x: self.predict_set_x[self.index:(self.index+1)]
                }
            )
    
    def connect(self):
        self.cidpress = self.fig.canvas.mpl_connect('key_press_event', self.press)
        self.work_on_spectrum(self.filelist[self.i])

    def press(self, event):
        c = event.key
        sys.stdout.flush()
        if c == 'n':
            print ''
            print 'keypress read:', c
            print 'going to next spectrum #', self.i
            if self.i < self.nspec:
                self.i += 1
                # if we have gone more than 100 spectra from the starting index, 
                # we need to read more data for the ANN to predict from
                if self.i >= self.start_iter+100:
                    datasets, size, wave_predict = load_SDSS_predict(None, self.i, self.i+100, self.filelist)
                    if self.filetype == 3:
                        self.predict_set_x, self.predict_set_y = datasets[0]
                    else:
                        # if not speclya files, only set_x is returned
                        self.predict_set_x = datasets
                    self.start_iter += 100
                    # self.predict_from_mlp = theano.function(
                    #     inputs=[self.index],
                    #     outputs=[self.classifier.y_pred, self.classifier.prediction_error_sq],
                    #     givens={
                    #         self.x: self.predict_set_x[self.index:(self.index+1)]
                    #     }
                    # )
                    self.predict_from_mlp = theano.function(
                        inputs=[self.index],
                        outputs=[
                            self.classifier.y_pred,
                            self.classifier.prediction_error_sq(self.y)
                        ],#hiddenLayer[0].output,
                        givens={
                            self.x: self.predict_set_x[self.index:(self.index+1)],
                            self.y: self.predict_set_y[self.index:(self.index+1)]
                        }
                    )
                self.work_on_spectrum(self.filelist[self.i])

            else:
                plt.close()
                print 'reached last spectrum in file'
                self.disconnect()
        elif c == 'b':
            print ''
            print 'keypress read:', c
            print 'going to last spectrum'
            if self.i > 0:
                self.i -= 1
                print self.i
                self.work_on_spectrum(self.filelist[self.i])
            else:
                self.work_on_spectrum(self.filelist[self.i])
        elif c == 'q':
            print ''
            print c, 'was pressed. Exit program.'
            plt.close()
            self.disconnect()
        else:
            print ''
            print 'neither n nor b was pressed. Display same spectrum again'
            self.work_on_spectrum(self.filelist[self.i])

    def work_on_spectrum(self, filename):
        # create spectrum object and determine properties of spectrum
        spec = spectrum()
        z_min = self.settings.z_min
        z_delta = self.settings.z_delta
        print 'filetype', self.filetype
        if self.filetype == 1:
            read_spSpec_fitsio(self.filelist[self.i].rstrip(), spec, None)
        if self.filetype == 2:
            read_spec_fitsio(self.filelist[self.i].rstrip(), spec, None)
        if self.filetype == 3:
            read_speclya_fitsio(self.filelist[self.i].rstrip(), spec, None)

        zem_index = calc_zem_index(spec.z, z_min, z_delta)
        print 'zem_index:', zem_index
        if zem_index == -1:
            zem_index = 0
        #x_ind, y_ind, y_err_ind = fit_powerlaw_individual(spec, self.settings, 1, zem_index = zem_index, deviation_factor=5.0)
        
        # clear plot from before and write labels
        self.axarr.clear()
        self.axarr.set_xlabel('Wavelength / Angstrom')
        self.axarr.set_ylabel('Flux')
        spec.wave = spec.wave / (1.0 + spec.z)

        #size = np.size(spec_median.powerlaw)
        # plot spectrum
        #flux_means = smooth_array(spec.flux, 25, spec.flux_error)
        # ly alpha:
        #ind = np.where(spec.wave/(1.0+spec.z) < 1200)[0]
        #flux_lya   = smooth_array(spec.flux[ind], 100, spec.flux_error[ind])
        #        flux_lya   = smooth_array(flux_lya[ind], 5, spec.flux_error[ind])

        self.axarr.plot(spec.wave, spec.flux, 'r-', linewidth=0.5)
        #self.axarr.plot(spec.wave, spec.model_sdss, 'g-')

        #axarr.plot(wave_predict[i], predict_set_x[i].eval(), 'r-')
        #axarr.plot(wave_predict[i], predict_set_y[i].eval(), 'g-')

        # before we plot, we need to revert the normalization done on the spectrum
        # before prediction
        #lya_low = 1041#*(spec.z + 1.0)
        #lya_up  = 1185#*(spec.z + 1.0)
        lya_low = 1030
        lya_up  = 1585
        index = np.where((spec.wave > lya_low) &
                         (spec.wave < lya_up ))[0]
        if index != []:
            regulator = np.percentile(spec.flux[index], 100)
        else:
            print 'regulator broken!!'
            regulator = 1
        #regulator = 1
        
        # dust correction related properties
        spec.Ebv = obstools.get_SFD_dust(spec.coordinates.galactic.l.deg, spec.coordinates.galactic.b.deg, self.dustmap, interpolate=0)
        spec.filename = filename
        Gal_extinction_correction(spec)



        predict_index = self.i - self.start_iter
        print 'predict_index', predict_index
        if self.filetype == 3:
            flux_prediction, prediction_error = self.predict_from_mlp(predict_index)
            flux_prediction = flux_prediction[0]
            # to compare prediction_error theano with numpy:
            # calc prediction error:
            model_y = self.predict_set_y[predict_index].eval()
            prediction_error_manual = (flux_prediction - model_y)**2
            print 'prediction error manually and theano', prediction_error_manual, prediction_error
            print 'max, percentile', np.max(prediction_error_manual), np.percentile(prediction_error_manual, 75)
        else:
            flux_prediction = self.predict_from_mlp(predict_index)
            flux_prediction = flux_prediction[0][0]

        # now revert the normalization
        flux_prediction *= regulator

        if np.size(flux_prediction) != np.size(self.wave_predict[0]):
            flux_prediction = rebin_1d(flux_prediction, np.size(self.wave_predict[0]), method='spline')
        ind_nonzero = np.nonzero(flux_prediction)
        
        # model_for_pred = theano.function(
        #     inputs=[self.index],
        #     outputs=self.predict_set_y[self.index],
        # )

        self.axarr.plot(self.wave_predict[predict_index], flux_prediction, 'm-')
        #self.axarr.plot(self.wave_predict[predict_index], model_for_pred(predict_index)*regulator, 'k-')
        if self.filetype == 3:
            self.axarr.plot(self.wave_predict[predict_index], model_y*regulator, 'k-')


        # flux corrected spectrum
        # plot spectrum error
        self.axarr.plot(spec.wave, spec.flux_error, 'b-')
        # plot fit_powerlaw_individual
        #self.axarr.plot(spec.wave, spec.powerlaw,'m-', linewidth=2.5)

        # set title
        filename   = str(spec.filename) + '    '
        zem_str    = 'zem: ' + "{0:.5f}".format(float(spec.z)) + '    ' + 'Alt: ' + '\n'#"{0:.4}".format(float(spec.altitude)) + '\n'
        individual = 'alpha_ind: ' + "{0:.5f}".format(spec.alpha) + '    ' + 'chisq_ind: ' + "{0:.5f}".format(spec.chisq)# + '\n'
        #median     = 'alpha_med: ' + "{0:.5f}".format(spec_median.alpha) + '    ' + 'chisq_med: ' + "{0:.5f}".format(spec_median.chisq)
        title = filename + zem_str + individual# + median

        self.axarr.set_title(title)
        self.axarr.set_xlim(900, 1600)
#        self.axarr.set_ylim(-1/4.0*np.max(spec.flux[50:-50]), np.max(spec.flux[50:-50]))
        print np.min(spec.flux), np.max(spec.flux)
        print np.where(spec.flux == np.max(spec.flux))[0]
        # show plot

        plt.draw()

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cidpress)


def main(args):

    if len(args) > 0:
        filelist = open(args[0], 'r').readlines()
    else:
        import sys
        sys.exit('No input file given!')
    if '--ann' in args:
        try:
            i = args.index('--ann')
            saved_ann_str = args[i+1]
        except IndexError:
            import sys
            sys.exit('If you provide --ann option, also need to supply a file')
    else:
        saved_ann_str = 'classifier_best_params.mlp'

    saved_ann = open(saved_ann_str)
    layer_params = cPickle.load(saved_ann)
    ann_layout   = cPickle.load(saved_ann)
    saved_ann.close()

    # create function to predict from classifier

    x = T.matrix('x')
    y = T.matrix('y')

    rng = np.random.RandomState(1234)

    # construct the MLP class
    if '--mlp' in args:
        classifier = MLP(
            rng=rng,
            input=x,
            layer_params=layer_params,
            ann_layout=ann_layout
        )
    
    # construct the LeNet class
    if '--lenet' in args:
        nkerns = [20, 50]
        batch_size = 1
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
            layer_params = layer_params
        )

    # construct the dA class
    if '--dA' in args:
        da = dA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=x,
            n_visible=44 * 44,
            n_hidden=500
        )        

    # determine if it's a DR7 or DR10 file
    filetype = check_filetype(filelist[0])
    print filetype

    dustmap = '/home/basti/SDSS_indie/dust_maps/maps/SFD_dust_4096_%s.fits'

    print 'This program plots the spectra from a list of FITS files'
    print 'as well as PCA model and ANN prediction'
    print 'Red:   spectrum'
    print 'Blue:  error of spectrum'
    print 'Pink:  fit_powerlaw_individual (based on individual pixels in intervals'
    print ''
    print 'press n to dispaly the next spectrum'
    print 'press b to display the last spectrum'


    # number of files in the list
    file_num = len(filelist)
    settings = program_settings()
    fig, axarr = plt.subplots(1, sharex=True, figsize=(10,8), dpi=100)
    # create WorkOnSpectrum object
    resid_corr = read_resid_corr('residcorr_v5_4_45.dat')
    spectra = WorkOnSpectrum(filelist, fig, axarr, dustmap, settings, resid_corr, classifier, x)
    spectra.connect()
    plt.show()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])


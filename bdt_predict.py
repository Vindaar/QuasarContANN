#!/usr/bin/env python

# This program plots the spectrum of a FITS file, fits the continuum
# using two different fitting algorithms
# 1: using medians of fitting intervals
# 2: using all data points

from copy import deepcopy
import sys
from SDSSmodules.SDSSfiles import *
from SDSSmodules.SDSSutilities import *
from SDSSmodules.SDSSfitting import *
from SDSSmodules.SDSSclasses import spectrum, program_settings
import numpy as np
#from astropy.io import FITS
import fitsio
from pylab import *
from matplotlib.backend_bases import Event
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from congrid import rebin_1d, get_arrays


# class which contains all necessary functions to work on the matplotlib graph
class WorkOnSpectrum:
    def __init__(self, filelist, figure, ax, filetype, dustmap, settings, resid_corr, clf):
        self.filelist  = filelist
        self.filetype  = filetype
        self.fig       = figure
        self.axarr     = ax
        self.dustmap   = dustmap
        self.nspec     = len(filelist)
        self.i         = 0
        self.settings  = settings
        self.resid_corr= resid_corr
        self.clf       = clf
    
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
        if self.filetype == 1:
            read_spSpec_fitsio(self.filelist[self.i].rstrip(), spec, None)
        if self.filetype == 2:
            read_spec_fitsio(self.filelist[self.i].rstrip(), spec, None)
        if self.filetype == 3:
            read_speclya_fitsio(self.filelist[self.i].rstrip(), spec, None)
        # dust correction related properties
        spec.Ebv = obstools.get_SFD_dust(spec.coordinates.galactic.l.deg, spec.coordinates.galactic.b.deg, self.dustmap, interpolate=0)
        spec.filename = filename
        Gal_extinction_correction(spec)
        # perform the fit. 1st with medians 2nd with individual points
                
        spec_median = deepcopy(spec)

        zem_index = calc_zem_index(spec.z, z_min, z_delta)
        if zem_index == -1:
            zem_index = 0
        x_ind, y_ind, y_err_ind = fit_powerlaw_individual(spec, self.settings, 1, zem_index = zem_index, deviation_factor=5.0)
        # clear plot from before and write labels
        self.axarr[0].clear()
        self.axarr[1].clear()
        self.axarr[0].set_xlabel('Wavelength / Angstrom')
        self.axarr[0].set_ylabel('Flux')

        fl, wa = get_arrays(spec)
        fl = rebin_1d(fl, 563, method='spline')
        wa = rebin_1d(wa, 563, method='spline')
        predict_model = self.clf.predict(fl)[0]

        spec.wave = spec.wave / (1.0 + spec.z)

        size = np.size(spec_median.powerlaw)
#        print np.shape(spec.wave), np.shape(spec_median.powerlaw[0:size])
        # plot spectrum
        ind = np.where(spec.wave/(1.0+spec.z) < 1200)[0]

        self.axarr[0].plot(spec.wave, spec.flux, 'r-', linewidth=0.5)
        # plot spectrum continuum PCA model
        if self.filetype == 3:
            self.axarr[0].plot(spec.wave, spec.model_sdss, 'g-')
        # flux corrected spectrum
        # plot spectrum error
        self.axarr[0].plot(spec.wave, spec.flux_error, 'b-')
        # plot fit_powerlaw_individual
        self.axarr[0].plot(spec.wave, spec.powerlaw,'m-', linewidth=2.5)
        # plot prediction
        self.axarr[0].plot(wa[:-1], predict_model[:-1], 'c-', linewidth=2.5)
        # set title
        filename   = str(spec.filename) + '    '
        zem_str    = 'zem: ' + "{0:.5f}".format(float(spec.z)) + '    ' + '\n'
        individual = 'alpha_ind: ' + "{0:.5f}".format(spec.alpha) + '    ' + 'chisq_ind: ' + "{0:.5f}".format(spec.chisq) + '\n'
        median     = 'alpha_med: ' + "{0:.5f}".format(spec_median.alpha) + '    ' + 'chisq_med: ' + "{0:.5f}".format(spec_median.chisq)
        title = filename + zem_str + individual + median

        self.axarr[0].set_title(title)
        self.axarr[0].set_xlim(850, 3000)
#        self.axarr[0].set_ylim(-1/4.0*np.max(spec.flux[50:-50]), np.max(spec.flux[50:-50]))
        print np.min(spec.flux), np.max(spec.flux)
        print np.where(spec.flux == np.max(spec.flux))[0]
        # show plot

        plt.draw()

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cidpress)


def main(args):

    if len(args) > 0:
        filelist = open(args[0], 'r').readlines()

    # determine if it's a DR7 or DR10 file
    filetype = check_filetype(filelist[0])

    dustmap = '/home/basti/SDSS_indie/dust_maps/maps/SFD_dust_4096_%s.fits'

    print 'This program plots the spectra from a list of FITS files'
    print 'Red:   spectrum'
    print 'Blue:  error of spectrum'
    print 'Pink:  fit_powerlaw_individual (based on individual pixels in intervals'
    print 'Green: fit_powerlaw            (based on medians in intervals)'
    print 'Cyan:  data points used for fit_powerlaw_individual'
    print 'Black: data points used for fit_powerlaw'
    print ''
    print 'press n to dispaly the next spectrum'
    print 'press b to display the last spectrum'

    begin = int(raw_input('Give the spectrum with which to begin: '))

    # number of files in the list
    file_num = len(filelist)
    settings = program_settings()
    fig, axarr = plt.subplots(2, sharex=True, figsize=(10,8), dpi=100)
    # create WorkOnSpectrum object
    resid_corr = read_resid_corr('residcorr_v5_4_45.dat')
    clf = joblib.load('DR9_tree.pkl')
    #print clf
    spectra = WorkOnSpectrum(filelist[begin:], fig, axarr, filetype, dustmap, settings, resid_corr, clf)
    spectra.connect()
    plt.show()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])


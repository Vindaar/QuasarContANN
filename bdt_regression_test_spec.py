#!/usr/bin/env python
"""
===================================================================
Decision Tree Regression
===================================================================

A 1D regression with decision tree.

The :ref:`decision trees <tree>` is
used to fit a sine curve with addition noisy observation. As a result, it
learns local linear regressions approximating the sine curve.

We can see that if the maximum depth of the tree (controlled by the
`max_depth` parameter) is set too high, the decision trees learn too fine
details of the training data and learn from the noise, i.e. they overfit.
"""
print(__doc__)

import numpy as np
import fitsio
from SDSSmodules.SDSSfiles import *
from SDSSmodules.SDSSclasses import *
from sklearn.externals import joblib

#SVM
# Fit regression model
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

#BDT
from sklearn.ensemble import AdaBoostRegressor
#from sklearn.metrics import accuracy_score
from congrid import rebin_1d, get_arrays

# Create a random dataset

files = open('../../dr9list.lis', 'r').readlines()

spec = spectrum()
#flux = np.empty((3,1))
flux = []
flux_error = []
model_sdss = []
size = []
wave = []
nspec = 10000
print nspec

for i in xrange(nspec):
    read_speclya_fitsio(files[i], spec)
    #flux = np.asarray([spec.flux[100:200]]).T
    wave_temp = spec.wave / (spec.z + 1.0)
    #print wave
    lya_low = 1041#*(spec.z + 1.0)
    lya_up  = 1185#*(spec.z + 1.0)
    # index = np.where((spec.flux_error > 0) &
    #                  (spec.wave > lya_low) &
    #                  (spec.wave < lya_up ) &
    #                  (spec.mask_comb == 0))[0]
    index = np.where((wave_temp > lya_low) &
                     (wave_temp < lya_up ))[0]
    size.append(np.size(index))
    print size[i]
    flux.append(spec.flux[index])
    #flux[i] = spec.flux[index]
    #wave = spec.wave[index]
    flux_error.append(spec.flux_error[index])
    model_sdss.append(spec.model_sdss[index])
    #wave = np.asarray([wave]).T
    wave.append(wave_temp[index])


max = int(max(size))
for i in xrange(nspec):
    flux[i] = rebin_1d(flux[i], max, method='spline')
    wave[i] = rebin_1d(wave[i], max, method='spline')
    model_sdss[i] = rebin_1d(model_sdss[i], max, method='spline')
    print np.shape(flux[i])

flux = np.asarray(flux)
wave = np.asarray(wave)
model_sdss = np.asarray(model_sdss)
print flux[0, :]
min_size = int(min(size))
flux_ar = np.empty((min_size, nspec))    
model_ar = np.empty((min_size, nspec))
# for i in xrange(nspec):
#     flux_ar[:,i] = flux[i][:min_size]
#     model_ar[:,i] = model_sdss[i][:min_size]


flux_error = np.asarray(flux_error)
rng = np.random.RandomState(1)
# X = np.sort(5 * rng.rand(80, 1), axis=0)
# y = np.sin(X).ravel()
# y[::5] += 3 * (0.5 - rng.rand(16))
# X = np.sort(5 * np.random.rand(40, 1), axis=0)
# y = np.sin(X).ravel()
#X = np.linspace(0, 6, 100)[:, np.newaxis]
#y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])
X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y[::5, :] += (0.5 - rng.rand(20, 2))


#flux_smooth = smooth_array(flux, 3)

#svr_rbf = SVR(kernel='rbf', C=0.5, gamma=0.2e-3)
svr_rbf = SVR(kernel='rbf', C=1e4, gamma=1e4)
#svr_rbf = SVR(kernel='rbf', C=50, gamma=0.2e-3)
#print np.shape(wave), np.shape(flux), np.shape(model_sdss)
#mean_snr = int(np.ma.average(flux / flux_error))
#mean_snr = min(mean_snr, 2)
#print mean_snr
#y_rbf = svr_rbf.fit(flux, model_sdss).predict(flux[2])#, sample_weight=flux[:,0]**(mean_snr)/flux_error).predict(flux)
#print y_rbf, np.shape(y_rbf)
#clf_1 = DecisionTreeRegressor(max_depth=500)

#clf_1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=100),
#                          n_estimators=300, random_state=rng)

clf_1 = DecisionTreeRegressor(max_depth=500)
#clf_2 = DecisionTreeRegressor(max_depth=500)
#clf_1.fit(flux_ar, model_ar)
clf_1.fit(flux, model_sdss)
#clf_2.fit(flux, wave)

joblib.dump(clf_1, 'DR9_tree.pkl')

# Predict
#X_test = spec.wave[100:200, np.newaxis]
#print np.shape(X_test)

read_speclya_fitsio(files[8314], spec)
fl, wa = get_arrays(spec)

fl = rebin_1d(fl, max, method='spline')
wa = rebin_1d(wa, max, method='spline')
print np.size(wa)

#y_rbf = svr_rbf.fit(flux_ar, model_ar).predict(fl[:min_size])#, sample_weight=flux[:,0]**(mean_snr)/flux_error).predict(flux)
y_1 = clf_1.predict(fl)[0]

#y_2 = clf_2.predict(X_test)

# Plot the results
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(2, sharex=True, figsize=(10,8), dpi=100)
#print np.size(wave[1]), np.size(y_1), np.shape(flux_ar[0])
#print y_1[0], wave[1]

spec.wave = spec.wave / (spec.z + 1.0)


#plt.figure()
#plt.scatter(wave, flux, c="r", label="data", marker='o')
axarr[0].plot(wa[:-1], y_1[:-1], c="b", label="max_depth=2", linewidth=2)

axarr[0].plot(spec.wave, spec.flux, 'r-')
axarr[0].plot(spec.wave, spec.flux_error, 'm-')
axarr[0].plot(spec.wave, spec.model_sdss, 'g-')
#axarr[0].plot(wave[:,0], y_rbf, c='g', label='RBF model', linewidth=2)
#axarr[0].plot(wave, y_rbf, c='b', label='RBF model', linewidth=2)

#plt.plot(X_test, y_2, c="r", label="max_depth=5", linewidth=2)
axarr[0].set_xlabel("data")
axarr[0].set_ylabel("target")
axarr[0].set_title("Decision Tree Regression")

axarr[0].set_xlim(850, 3000)
#axarr[1].plot(flux, model_sdss, 'g-', linewidth=0.5)
#axarr[1].plot(wave, flux/spec.model_sdss[index], 'b-', linewidth=0.5)
plt.legend()
plt.show()


print 'Maximal array length was: ', max

#!/usr/bin/env python
from sklearn.externals import joblib

model_fname = 'model.pkl'
clf = joblib.load(model_fname)

print "Loaded classifier {}".format(clf)

print "Intercept = {}\nCoefs = {}".format(clf.intercept_, clf.coef_)

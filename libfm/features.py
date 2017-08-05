from __future__ import division
import math
import logging
import bisect

from baskets import constants

all_features = set()
class FeatureMeta(type):
  """We want Features to basically just be functions with some metadata. This
  metaclass makes that easy by magically turning their methods into class
  methods, and overriding the effect of calling the class (so that it actually
  returns the result of 'calling' the feature function, rather than trying to
  instantiate a feature object).
  """
  wrapped_classmethods = ['call', 'get_relative_index', 'get_value']
  def __new__(meta, name, parents, dct):
    dct.setdefault('name', name)
    for methodname in meta.wrapped_classmethods:
      if methodname in dct:
        dct[methodname] = classmethod(dct[methodname])
    for attr_name, val in dct.iteritems():
      if callable(val) and attr_name not in meta.wrapped_classmethods and attr_name != '__metaclass__':
        logging.warn("Don't recognize method {}. Not wrapping it.".format(attr_name))
    return super(FeatureMeta, meta).__new__(meta, name, parents, dct)

  def __init__(cls, name, parents, dct):
    if not hasattr(cls, 'size') and issubclass(cls, BucketizedFeature):
      setattr(cls, 'size', len(cls.boundaries)+1)
    if 'abstract' not in dct:
      assert hasattr(cls, 'size'), "Concrete feature {} has no defined size".format(name)
      all_features.add(cls)
    return super(FeatureMeta, cls).__init__(name, parents, dct)

  # (Can't do this at the class level. In that case, "calling" the class will
  # still just result in instantiating it.)
  def __call__(cls, *args, **kwargs):
    return cls.call(*args, **kwargs)

class Feature(object):
  abstract = True
  __metaclass__ = FeatureMeta
  default = True
  def call(kls, example):
    msg = '{} must implement call() method'.format(kls.name)
    raise NotImplementedError(msg)

class UnaryFeature(Feature):
  abstract = True
  def call(self, example):
    i = self.get_relative_index(example)
    val = self.get_value(example)
    yield i, val

class ScalarFeature(UnaryFeature):
  abstract = True
  size = 1
  def get_relative_index(kls, example):
    return 0

class OneHotFeature(UnaryFeature):
  abstract = True
  def get_value(kls, example):
    return 1

# i.e. not a UnaryFeature
class MultiFeature(Feature):
  abstract = True

INCLUDE_FOCAL_PID_IN_OTHERPROD_FEATS = False
class OtherProdFeature(MultiFeature):
  abstract = True
  size = constants.N_PRODUCTS

class BucketizedFeature(OneHotFeature):
  abstract = True
  # Concrete subclasses must have a boundaries attribute e.g.
  # boundaries = [0, 1, 2]
  # Buckets include left boundaries and exclude right boundaries. The above 
  # example leads to buckets (-inf, 0), [0, 1), [1, 2), [2, inf)

  # (size is set by metaclass after class instantiation, based on boundaries)
  def get_orig_value(kls, example):
    raise NotImplementedError

  def get_relative_index(kls, example):
    return bisect.bisect_right(kls.boundaries)

##### Concrete features

class DaysSincePrior_Bucketized(BucketizedFeature):
  boundaries = [1, 3, 7, 14, 30]
  def get_orig_value(kls, example):
    return example.gfs['days_since_prior']

class Hour(OneHotFeature):
  size = 24
  def get_relative_index(kls, example):
    return example.gfs['hour']

class Dow(OneHotFeature):
  size = 7
  def get_relative_index(kls, example):
    return example.gfs['dow']

class Uid(OneHotFeature):
  size = constants.N_USERS
  def get_relative_index(kls, example):
    return example.gfs['uid']

class DaysSincePrior(ScalarFeature):
  default = False
  def get_value(kls, example):
    days = example.gfs['days_since_prior']
    return days / 30

class FocalPid(OneHotFeature):
  size = constants.N_PRODUCTS
  def get_relative_index(kls, example):
    return example.pid - 1

# TODO: transformations chosen here are pretty handwavey.
# Might just want to bucketize instead?
class FocalLogFrequency(ScalarFeature):
  def get_value(kls, example):
    freq = example.pfs[example.pid]['frequency']
    return math.log(freq)

class FocalRecencyDays(ScalarFeature):
  def get_value(kls, example):
    rec = example.pfs[example.pid]['recency_days']
    return 4 / (4+rec)

class FocalRecencyOrders(ScalarFeature):
  def get_value(kls, example):
    rec = example.pfs[example.pid]['recency_orders']
    return 2 / (2+rec)

class OtherProdLogFrequency(OtherProdFeature):
  def call(kls, example):
    # TODO: could maybe refactor out some common logic here
    for pid in example.pfs:
      if not INCLUDE_FOCAL_PID_IN_OTHERPROD_FEATS and pid == example.pid:
        continue
      i = pid-1
      freq = example.pfs[pid]['frequency']
      val = math.log(freq)
      yield i, val

class OtherProdPid(OtherProdFeature):
  def call(kls, example):
    for pid in example.pfs:
      if not INCLUDE_FOCAL_PID_IN_OTHERPROD_FEATS and pid == example.pid:
        continue
      i = pid-1
      val = 1
      yield i, val


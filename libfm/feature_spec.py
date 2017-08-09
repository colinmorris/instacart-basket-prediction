
import features

class FeatureSpec(object):
  def __init__(self, feats):
    self.features = self.sort_features(feats)

  def add_feature(self, feat):
    self.features.append(feat)
    self.features = self.sort_features(self.features)

  @classmethod
  def sort_features(kls, feats):
    scalars = [feat for feat in feats if issubclass(feat, features.ScalarFeature)]
    unaries = [feat for feat in feats if issubclass(feat, features.UnaryFeature) 
        and feat not in scalars]
    the_rest = [feat for feat in feats if feat not in scalars and feat not in unaries]
    groups = [scalars, unaries, the_rest]
    for featgroup in groups:
      featgroup.sort(key=lambda f: f.name)
    return sum(groups, [])

  @classmethod
  def all_features_spec(kls):
    return kls(features.all_features)

  @classmethod
  def feature_spec_excepting(kls, omit):
    feats = features.all_features.difference(omit)
    return kls(feats)

  @classmethod
  def barebones_spec(kls):
    feats = [f for f in features.all_features if f.size < 1000 and f.default]
    return kls(feats)

  @classmethod
  def basic_spec(kls):
    feats = [f for f in features.all_features if not issubclass(f, features.OtherProdFeature)
        and f.default
        ]
    return kls(feats)

  def make_featdict(self, example):
    fd = {}
    offset = 0
    for feature in self.features:
      for rel_idx, val in feature(example):
        #assert rel_idx < feature.size
        fd[offset+rel_idx] = val
      offset += feature.size
    return fd

  def _lookup_global_index(self, i):
    offset = 0
    for feature in self.features:
      offset += feature.size
      if i < offset:
        rel_idx = i - (offset - feature.size)
        return rel_idx, feature
    assert False

  def parse_libfm_line(self, line):
    featstr = line[2:]
    featstrs = featstr.split(' ')
    fd = {}
    for feat_nugget in featstrs:
      id, val = feat_nugget.split(':')
      id = int(id)
      val = float(val) if '.' in val else int(val)
      assert id not in fd
      fd[id] = val
    return self.parse_featdict(fd)

  # TODO: this is not currently going to handle id/onehottish features
  # very well. But I'm not really using it, so fine.
  def parse_featdict(self, fd):
    parsed = {}
    for global_idx, val in fd.iteritems():
      i, feat = self._lookup_global_index(global_idx)
      if issubclass(feat, features.MultiFeature):
        extant = parsed.get(feat.name, {})
        extant[i] = val
      elif issubclass(feat, features.OneHotFeature):
        assert val == 1, "Expected onehot feature {} to have value 1 but was {}".format(
            feat.name, val)
        assert feat.name not in parsed
        parsed[feat.name] = i
      elif issubclass(feat, features.ScalarFeature):
        assert i == 0, "Expected relative idx of 0 for scalar feature {}, got {}".format(
            feat.name, i)
        assert feat.name not in parsed
        parsed[feat.name] = val
      else:
        assert False, "Feature {} has unrecognized type".format(feat.name)

    return parsed

  def write_group_file(self, f):
    """For grouping, libfm wants a text file with as many lines as features in the
    dataset, where the ith line gets k, where feature_i \in group_k
    """
    for group_idx, feat in enumerate(self.features):
      # TODO: Not sure if it makes sense to have each scalar feature be its own
      # 'group', or to just throw them together. It's possible all the scalar 
      # features are just a bad idea, and should all be replaced with bucketized versions.
      groupstr = '\n'.join(str(group_idx) for _ in range(feat.size)) + '\n'
      f.write(groupstr)

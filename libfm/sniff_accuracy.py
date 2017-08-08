"""
Written while debugging some odd accuracy numbers I was seeing during training
(Turned out to be legitimately odd accuracy resulting from leaking some 
forbidden info through a badly implemented feature)
"""
from __future__ import division
from collections import defaultdict

def get_labels(fname):
  f = open(fname)
  labels = []
  for line in f:
    label = int(line[0])
    labels.append(label)
  return labels

prob_fname = 'predictions.out'
with open(prob_fname) as f:
  probs = map(float, f.readlines())
vec_fname = 'vectors/test.libfm'

labels = get_labels(vec_fname)


assert len(probs) == len(labels)

res = defaultdict(int)

for prob, label in zip(probs, labels):
  pred = prob >= .5
  res['tp'] += pred and label
  res['fn'] += not pred and label
  res['fp'] += pred and not label
  res['tn'] += not pred and not label

print res
right = res['tp'] + res['tn']
wrong = res['fn'] + res['fp']
total = right + wrong

print '{} right out of {} (acc = {:.3%}'.format(right, total, right/(right+wrong))
nneg = res['tn'] + res['fp']
print 'Baseline accuracy = {:.3%}'.format(nneg / total)

print 'Macro-fscore = {:.3%}'.format(2*res['tp'] / (2*res['tp'] + wrong))

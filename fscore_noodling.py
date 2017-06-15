from __future__ import division
from scipy.special import binom

DEBUG = 0

def dbg(msg):
    if DEBUG:
        print msg

def expected_fscore(n_predicted, n=100, prob_per_instance=.01):
    #assert n_predicted in (1, n)
    ppi = prob_per_instance
    ppibar = (1 - prob_per_instance)
    def fscore(fn, tp, fp):
        num = 2*tp
        denom = 2*tp + fp + fn
        return num / denom

    parts = []
    for ntrue in range(1, n+1):
        # ntrue := this many were actually true in the gold standard
        if n_predicted == 1:
            tp = 1
            fn = ntrue - 1
            fp = 0
            # Prob. of our 1 predicted label being true
            prob_weight = ppi
            # Prob of ntrue-1 other labels being true out of n-1
            prob_weight *= binom(n-1, ntrue-1) * ppi**(ntrue-1) * ppibar**(n-ntrue)
        elif n_predicted == n:
            tp = ntrue
            fn = 0
            fp = n - ntrue
            prob_weight = binom(n, ntrue) * ppi**ntrue * ppibar**(n-ntrue)
        else:
            assert False, "fuck it"
            for tp in range(1, ntrue):
                fp = n_predicted - tp
                fn = ntrue - tp


        fs = fscore(fn, tp, fp)
        dbg('fscore = {}'.format(fs))
        dbg('prob weight = {}'.format(prob_weight))
        res = fs * prob_weight
        parts.append(res)
    return parts

ef = expected_fscore

foo = ef(1, 2, 1/4)

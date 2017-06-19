from __future__ import division

import numpy as np
import pandas as pd

FEATURE_COLS = [
        'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order',
        'prev_prods', 'prev_reorders'
]

class Dataset(object):

    def __init__(self, df, hps):
        self.seqids = df['seqid'].unique()
        np.random.shuffle(self.seqids)
        self.batch_size = hps.batch_size
        self.n = len(df)
        self.nbatches = self.n // self.batch_size
        labels, vecs, seqlens = self.convert_df(df, hps.max_seq_len)
        self.labels = labels
        self.vectors = vecs
        self.seqlens = seqlens

    @classmethod
    def load(kls, hps, fname='vectors.pickle'):
        df = pd.read_pickle(fname)
        return Dataset(df, hps)

    def convert_df(self, df, maxlen):
        """Return ndarrays of labels, vectors, and seqlens. Or something."""
        #self.labels = df['reordered'] # TODO: convert to int8 or whatever?
        nseqs = len(self.seqids)
        labels = np.zeros( (nseqs, maxlen), dtype=np.bool_)
        seqlens = np.zeros(nseqs, dtype=np.int32)
        nfeats = len(FEATURE_COLS)
        vectors = np.zeros( (nseqs, maxlen, nfeats), dtype=np.float32)
        i = 0
        for (seqid, group) in df.groupby('seqid'):
            slen = len(group)
            seqlens[i] = slen
            labels[i,:slen] = group['reordered']
            vectors[i,:slen] = group.loc[:,FEATURE_COLS]
        return (labels, vectors, seqlens)

    def get_batch(self, idx):
        #assert idx < self.nbatches
        idx = idx % self.nbatches
        a = idx * self.batch_size
        b = (idx+1) * self.batch_size
        slice = np.s_[a:b]
        return (
                self.labels[slice],
                self.vectors[slice],
                self.seqlens[slice],
        )

from __future__ import division

import numpy as np
import pandas as pd

FEATURE_COLS = [
        'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order',
        'prev_prods', 'prev_reorders'
]

class Dataset(object):
    NFEATS = len(FEATURE_COLS)
    TEST_PATH = 'testset.pickle'

    def __init__(self, df, hps):
        self.hps = hps # (not sure)
        self.batch_size = hps.batch_size
        self.nsteps = len(df)
        labels, vecs, seqlens, _ = self.convert_df(df, hps.max_seq_len)
        self.n = labels.shape[0]
        self.nbatches = self.n // self.batch_size
        self.labels = labels
        self.vectors = vecs
        self.seqlens = seqlens

    @classmethod
    def load(kls, hps, fname='vectors.pickle'):
        df = pd.read_pickle(fname)
        return Dataset(df, hps)

    @staticmethod
    def convert_df(df, maxlen):
        """Return ndarrays of labels, vectors, and seqlens. Or something."""
        seqids = df['seqid'].unique()
        nseqs = len(seqids)
        labels = np.zeros( (nseqs, maxlen), dtype=np.bool_)
        seqlens = np.zeros(nseqs, dtype=np.int32)
        pids = np.zeros(nseqs, dtype=np.int32)
        nfeats = len(FEATURE_COLS)
        vectors = np.zeros( (nseqs, maxlen, nfeats), dtype=np.float32)
        i = 0
        for (seqid, group) in df.groupby('seqid'):
            pids[i] = group.iloc[0]['prodid'] # Should be the same throughout group
            slen = len(group)
            seqlens[i] = slen
            labels[i,:slen] = group['reordered']
            vectors[i,:slen] = group.loc[:,FEATURE_COLS]
        return (labels, vectors, seqlens, pids)

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

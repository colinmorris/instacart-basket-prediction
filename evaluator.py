from __future__ import division
import numpy as np

class Results(object):

    def __init__(self, fscores):
        self.fscores = fscores

    def __repr__(self):
        return 'Results over {:,} orders with mean fscore = {:.4f}'.format(
                len(self.fscores),
                self.fscores.mean()
        )

class BaseEvaluator(object):

    def __init__(self, db):
        self.db = db

    def evaluate(self, predictor):
        raise NotImplemented()

    def fscore(self, actual, predicted):
        assert len(predicted) > 0, "Must predict at least one product\
                (may include 'none' product with pid=0)"
        if len(actual) == 0:
            # Use a dummy product_id of 0 as "None"
            actual = [0]
        actual = set(actual)
        predicted = set(predicted)
        tpos = len(actual.intersection(predicted))

        precision = tpos / len(predicted)
        recall = tpos / len(actual)
        if precision == recall == 0:
            return 0
        return 2 * (precision*recall) / (precision + recall)


class TrainEvaluator(BaseEvaluator):
    
    def prediction_fscore(self, order, predicted):
        ops = self.db.ops_train
        actual = ops.loc[
                (ops['order_id'] == order.order_id)
                & (ops['reordered'] == 1)
                , 'product_id']
        return self.fscore(actual, predicted)

    def evaluate(self, predictor, n=None):
        orders = self.db.orders\
            [self.db.orders['eval_set']=='train']
        if n is not None:
            orders = orders[:n]
        fscores = np.zeros(orders.shape)
        for i in range(len(orders)):
            order = orders.iloc[i]
            predicted = predictor.predict_order(order)
            score = self.prediction_fscore(order, predicted)
            fscores[i] = score
        return Results(fscores)


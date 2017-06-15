import pandas as pd

class BasePredictor(object):

    def __init__(self, db):
        self.db = db

    def predict_order(self, order):
        """Return list of product_id"""
        raise NotImplemented

class PreviousOrderPredictor(BasePredictor):

    def __init__(self, db, always_none=0):
        super(PreviousOrderPredictor, self).__init__(db)
        self.always_none = always_none

    def predict_order(self, order):
        """Return a series of product ids"""
        uid = order.user_id
        onum = order.order_number
        assert onum > 1, onum
        orders = self.db.orders
        prevorder = orders[
                (orders['user_id'] == uid)
                & (orders['order_number'] == onum-1)
                ]
        assert len(prevorder) == 1
        prevorder = prevorder.iloc[0]
        prevprods = self.db.get_ops(prevorder)
        prods = prevprods['product_id']
        if self.always_none:
            prods = prods.append(pd.Series([0]))
        return prods

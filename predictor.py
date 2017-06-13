
class BasePredictor(object):

    def __init__(self, db):
        self.db = db

    def predict_order(self, order):
        """Return list of product_id"""
        raise NotImplemented

class PreviousOrderPredictor(BasePredictor):

    def predict_order(self, order):
        uid = order.user_id
        onum = order.order_number
        assert onum > 1, onum
        orders = self.db.orders
        prevorder = orders[
                (orders['user_id'] == uid)
                & (orders['order_number'] == onum-1)
                ]
        prevprods = self.db.get_ops(prevorder)
        return [prod.product_id for prod in prevprods]


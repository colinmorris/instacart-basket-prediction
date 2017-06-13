
class BasketDB(object):

    def __init__(self, **tables):
        for (name, table) in tables.iteritems():
            setattr(self, name, table)

    def get_ops(self, order):
        """ops = orderered products"""
        assert order.eval_set != 'test'
        assert order.eval_set in ('prior', 'train')
        ops = self.ops_train if order.eval_set == 'train' else self.ops_prior
        return ops[ops['order_id'] == order.order_id]


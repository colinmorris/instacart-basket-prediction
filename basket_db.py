import numpy as np

class BasketDB(object):

    def __init__(self, **tables):
        for (name, table) in tables.iteritems():
            setattr(self, name, table)

    def get_userids(self, n=None):
        users = self.orders['user_id'].unique()
        if n is not None:
            users = users[:n]
        return users

    def get_ops(self, order):
        """ops = orderered products
        order is a row (i.e. series) from orders df 
        return a df, a subset of one of the ordered_products tables"""
        assert order.eval_set != 'test'
        assert order.eval_set in ('prior', 'train')
        ops = self.ops_train if order.eval_set == 'train' else self.ops_prior
        return ops[ops['order_id'] == order.order_id]

    def snoop(self, uid=None):
        """Snoop a random user's orders
        """
        lastn = 10
        orders = self.orders
        if uid is None:
            uids = orders.loc[orders['eval_set']=='train', 'user_id'].unique()
            uid = np.random.choice(uids)
        their_orders = orders[ (orders['user_id']==uid) ]\
                .sort_values('order_number', ascending=0)\
                ['order_id']\
                .iloc[-10:]\
                .map(lambda oid: self.orderobjify(oid))
        return Peek(uid, their_orders)
        


    def orderobjify(self, oid):
        order = self.orders[self.orders['order_id']==oid].iloc[0]
        ops = self.get_ops(order)
        # hack
        idxs = ops['product_id'] - 1
        prods = self.products.loc[idxs]
        return Order(order, prods)

class Order(object):
    PROD_DELIM = '\n    '
    def __init__(self, order_row, products):
        self.order = order_row
        self.products = products

    def product_repr(self, prod):
        return '{}<{}>'.format(prod.product_name, prod.product_id)

    def __repr__(self):
        prodstr = self.PROD_DELIM.join(
                [self.product_repr(p) for (_, p) in 
                    self.products.sort_values('product_name').iterrows()
                ]
                )
        return 'Order {} (+{} days, @{}:00):{}'.format(
                self.order.order_number, 
                self.order.days_since_prior_order,
                self.order.order_hour_of_day,
                self.PROD_DELIM + prodstr
        )

        
class Peek(object):

    def __init__(self, uid, orders):
        self.uid = uid
        self.orders = orders
        self.n = len(orders)

    def __str__(self):
        orderstr = '\n  '.join(str(orde) for orde in self.orders)
        return "Peek at user {}'s last {} orders:\n{}".format(
                self.uid, self.n, orderstr
                )

    def __repr__(self):
        return self.__str__()


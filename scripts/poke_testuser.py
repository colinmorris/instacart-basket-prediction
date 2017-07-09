import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from insta_pb2 import User, Order

f = open('testuser.pb')
ustr = f.read()
u = User()
u.ParseFromString(ustr)

o = u.orders[0]
print o
print list(o.products)

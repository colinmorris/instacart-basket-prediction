from basket_db import BasketDB
import time

def main():
  t0 = time.time()
  db = BasketDB.load(usecache=False)
  t1 = time.time()
  print "Loaded db with indexes in {:.1f}s".format(t1-t0)
  db.save()

if __name__ == '__main__':
  main()

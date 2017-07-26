userdir = ../dat/user_pbs
vectordir = ../dat/scalar_vectors

all:
	echo "nothing to see here"

vectors: $(vectordir)/validation.npy $(vectordir)/train.npy $(vectordir)/test.npy

train: $(vectordir)/train.npy
$(vectordir)/train.npy: $(userdir)/train.tfrecords scalar_vectorize.py
	./scalar_vectorize.py --max-prods 5 --outname train_5 train
	./scalar_vectorize.py train

validation: $(vectordir)/validation.npy
$(vectordir)/validation.npy: $(userdir)/validation.tfrecords scalar_vectorize.py
	./scalar_vectorize.py validation

test: $(vectordir)/test.npy
$(vectordir)/test.npy: $(userdir)/test.tfrecords scalar_vectorize.py
	./scalar_vectorize.py test

clean: 
	rm $(vectordir)/*.npy

userdir = dat/user_pbs
vectordir = dat/vectors

all:
	echo "nothing to see here"

vectors: $(vectordir)/validation.tfrecords $(vectordir)/train.tfrecords $(vectordir)/test.tfrecords $(vectordir)/unit_tests.tfrecords

$(userdir)/validation.tfrecords $(userdir)/test.tfrecords $(userdir)/train.tfrecords: $(userdir)/users.tfrecords
	python preprocessing/partition_users.py --traintest

$(vectordir)/train.tfrecords: $(userdir)/train.tfrecords
	vectorize.py --max-prods 5 train

$(vectordir)/validation.tfrecords: $(userdir)/validation.tfrecords
	vectorize.py --max-prods 5 validation

$(vectordir)/test.tfrecords: $(userdir)/test.tfrecords
	vectorize.py test

$(vectordir)/unit_tests.tfrecords: $(userdir)/train.tfrecords
	vectorize.py -n 50 --max-prods 5 --out unit_tests train

clean: 
	rm $(vectordir)/*.tfrecords
	rm $(userdir)/train.tfrecords
	rm $(userdir)/test.tfrecords
	rm $(userdir)/validation.tfrecords

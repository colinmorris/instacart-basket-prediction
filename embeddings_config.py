"""
To enable the embedding visualization stuff in TensorBoard, run this and then copy
the contents of the checkpoint dir for your model to the corresponding log dir.
# TODO: would be cool to add some more metadata so that e.g. you could colour
  the products by aisle/dept in the embedding viz.
"""

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import csv
import sys

tag = sys.argv[1]

EMBS = [
    ('product', 'products.csv'),
    ('aisle', 'aisles.csv'),
    ('dept', 'departments.csv'),
]

def copy_embedding_metadata(name, csv_fname):
  f_csv = open('dat/'+csv_fname)
  fout = open('logs/{}/{}.tsv'.format(tag, name), 'w')
  reader = csv.reader(f_csv)
  reader.next() # Skip header
  for row in reader:
    thing = row[1]
    fout.write(thing+'\n')
  f_csv.close()
  fout.close()

config = projector.ProjectorConfig()

for (name, csv_name) in EMBS:
  copy_embedding_metadata(name, csv_name)
  emb = config.embeddings.add()
  emb.tensor_name = 'instarnn/{}_embeddings'.format(name)
  emb.metadata_path = '{}.tsv'.format(name)

summ_writer = tf.summary.FileWriter('logs/{}'.format(tag))
projector.visualize_embeddings(summ_writer, config)

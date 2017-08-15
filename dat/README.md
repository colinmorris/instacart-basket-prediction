Data files go here.

- `csv/`: raw csv files provided by Kaggle
- `pdicts/`: models' predicted probabilities for some user fold, as serialized dictionaries
- `scalar_vectors/`: recarrays saved as npy files, used by xgboost model
- `user_pbs/`: user protobufs (see `../preprocessing/`) 
- `vectors/`: tfrecords files with `SequenceExample`s used by RNN model
- `testuser.pb`: our canonical 'test user' (used in lots of unit tests) in protobuf form

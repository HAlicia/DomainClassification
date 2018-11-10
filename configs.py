import os

work_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(work_dir, 'data')
processed_data_dir = os.path.join(data_dir, 'processed_data')
embedding_dir = os.path.join(work_dir, 'embedding')

# datafile
raw_data = os.path.join(data_dir,  'train1029_w_bc.tsv')

# processed demo data path
train_datafile = os.path.join(processed_data_dir,  'train.csv')
val_datafile = os.path.join(processed_data_dir,  'val.csv')
test_datafile = os.path.join(processed_data_dir,  'test.csv')

result_dir = os.path.join(work_dir, 'Result')
svm_result_file = os.path.join(result_dir, 'SVM', 'svm_res.md')
# embedding file


SEED = 20181111
# params
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100

# hyper-params
# TODO
EPOCH_NUM = 25
BATCH_SIZE = 128
EMBEDDING_DIM = 300

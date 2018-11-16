import os

work_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(work_dir, 'data')
processed_data_dir = os.path.join(data_dir, 'processed_data')
embedding_dir = os.path.join(work_dir, 'embedding')
plot_dir = os.path.join(work_dir, 'plot')

# datafile
raw_data = os.path.join(data_dir, 'train_and_val.csv')
raw_data_cut = os.path.join(data_dir, 'train_and_val_cut.csv')

test_datafile = os.path.join(data_dir, 'test.csv')
test_datafile_cut = os.path.join(data_dir, 'test_cut.csv')

# TODO: processed data path
train_datafile = os.path.join(processed_data_dir, 'train.csv')
val_datafile = os.path.join(processed_data_dir, 'val.csv')

result_dir = os.path.join(work_dir, 'Result')
svm_result_file = os.path.join(result_dir, 'SVM', 'svm_res.md')

corpus_cut_token= os.path.join(data_dir, 'corpus_cut.txt')

# embedding file
glove100_path = os.path.join(embedding_dir, 'glove_mincnt5_win5_size100.txt')
fasttxt100_path = os.path.join(embedding_dir, 'ft_mincnt5_win5_size100.txt')
sg100_path = os.path.join(embedding_dir, 'sg_mincnt5_win5_size100.txt')
cbow100_path = os.path.join(embedding_dir, 'cbow_mincnt5_win5_size100.txt')

modelCheckpoint_dir = os.path.join(work_dir, 'model_dir')
log_dir = os.path.join(work_dir, 'log_dir')


singer_dict = os.path.join(data_dir, 'singer_name_list.txt')
song_dict = os.path.join(data_dir, 'song_name_list.txt')

SEED = 20181111
# params
MAX_NUM_WORDS = None
MAX_SEQUENCE_LENGTH = None

# hyper-params
# TODO

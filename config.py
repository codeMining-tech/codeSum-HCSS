import torch
import os
import time
import logging


# paths
datafolder ='preprocess_filterapi'
dataset_dir = '/data/'+datafolder
if not os.path.exists(dataset_dir):
    raise Exception('Dataset directory not exist.')

train_code_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'train.codes.json')
train_type_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'train.graph_nodes.json')
train_ast_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'train.nodes.json')
#train_tree_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'train.graph_nodes.json')
train_nl_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'train.nl.json')


valid_code_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'valid.codes.json')
valid_ast_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'valid.nodes.json')
valid_type_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'valid.graph_nodes.json')
valid_nl_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'valid.nl.json')
valid_edge_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'valid.edge.pkl')
valid_tree_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'valid.tree.pkl')

test_code_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'test.codes.json')
test_ast_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'test.nodes.json')
test_type_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'test.graph_nodes.json')
test_nl_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'test.nl.json')
test_edge_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'test.edge.pkl')
test_tree_path = os.getcwd()+'/'+os.path.join(dataset_dir, 'test.tree.pkl')


model_dir = os.getcwd()+'/data/model/'
best_model_path = 'best_model.pt'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

vocab_dir = os.getcwd()+ '/data/vocab/'
code_vocab_path = 'code_vocab.pk'
type_vocab_path = 'type_vocab.pk'
nl_vocab_path = 'nl_vocab.pk'
tree_vocab_path = 'tree_vocab.pk'

code_vocab_txt_path = 'code_vocab.txt'
type_vocab_txt_path = 'type_vocab.txt'
nl_vocab_txt_path = 'nl_vocab.txt'
tree_vocab_txt_path = 'tree_vocab.txt'

if not os.path.exists(vocab_dir):
    os.makedirs(vocab_dir)

out_dir = os.getcwd()+'/data/out/'    # other outputs dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# logger
log_dir = os.getcwd()+'/'+'data/log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

handler = logging.FileHandler(os.path.join(log_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime())) + '.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# device
use_cuda = False
#use_cuda = torch.cuda.is_available()
#device = torch.device('cuda' if use_cuda else 'cpu')
device = 'cpu'


# features
trim_vocab_min_count = False
trim_vocab_max_size = True


use_teacher_forcing = True
#use_check_point = False
use_check_point = True
use_lr_decay = True
use_early_stopping = True
#use_pointer_gen = False #
use_pointer_gen = False #

validate_during_train = True
save_valid_model = True
save_best_model = True
save_test_details = True
dropout = 0.2

nfeat = 512
nhid = 128
alpha = 0.2
nheads =4

# limitations
max_code_length = 400
max_type_length = 200
max_tree_length = 400
max_node_length = 20
max_nl_length = 30
min_nl_length = 3
max_decode_steps = 30
early_stopping_patience = 20


# hyperparameters
vocab_min_count = 5
type_vocab_size = 50000
tree_vocab_size = 50000
code_vocab_size = 50000
nl_vocab_size = 30000


embedding_dim = 512
hidden_size = 512
decoder_dropout_rate = 0.2
teacher_forcing_ratio = 0.5
batch_size = 1#32     # 128
epoch = 50     # 128
gat_encoder_lr = 0.001
code_encoder_lr = 0.001
tree_encoder_lr = 0.001
decoder_lr = 0.001
lr_decay_every = 1
lr_decay_rate = 0.99
n_epochs = 50    # 50

beam_width = 4
beam_top_sentences = 1     # number of sentences beam decoder decode for one input, must be 1 (eval.translate_indices)
eval_batch_size = 32    # 16
test_batch_size = 32

init_uniform_mag = 0.02
init_normal_std = 1e-4
eps = 1e-12


# visualization and resumes
print_every = 200   # 1000
plot_every = 1000     # 100
save_model_every = 6000   # 2000
save_check_point_every = 1000
validate_every = 6000     # 2000

# save config to log
save_config = True

config_be_saved = ['dataset_dir', 'use_cuda', 'device','use_teacher_forcing',
                   'use_lr_decay', 'use_early_stopping', 'max_code_length', 'max_type_length', 'max_nl_length', 'min_nl_length',
                   'max_decode_steps', 'early_stopping_patience']

train_config_be_saved = ['embedding_dim', 'hidden_size', 'decoder_dropout_rate', 'teacher_forcing_ratio',
                         'batch_size','gat_encoder_lr','code_encoder_lr',
                         'decoder_lr', 'lr_decay_every', 'lr_decay_rate', 'n_epochs']

eval_config_be_saved = ['beam_width', 'beam_top_sentences', 'eval_batch_size', 'test_batch_size']

if save_config:
    config_dict = locals()
    logger.info('Configurations this run are shown below.')
    logger.info('Notes: If only runs test, the model configurations shown above is not ' +
                'the configurations of the model test runs on.')
    logger.info('')
    logger.info('Features and limitations:')
    for config in config_be_saved:
        logger.info('{}: {}'.format(config, config_dict[config]))
    logger.info('')
    logger.info('Train configurations:')
    for config in train_config_be_saved:
        logger.info('{}: {}'.format(config, config_dict[config]))
    logger.info('')
    logger.info('Eval and test configurations:')
    for config in eval_config_be_saved:
        logger.info('{}: {}'.format(config, config_dict[config]))
    logger.info('')



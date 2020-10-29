import os
import random
import logging
# import basic python packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score

# import torch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from deepec.process_data import read_EC_Fasta, \
                                getExplainedEC_short, \
                                convertECtoLevel3
from deepec.data_loader import ECDataset, ECEmbedDataset
from deepec.utils import argument_parser, draw, save_losses, FocalLoss
from deepec.train import train, evalulate, train_emb, evalulate_emb
from deepec.model import DeepECv2_3, DeepECv2_4, DeepECv2_5, DeepEC_emb, TransformerModel, HybridModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')


class DeepECConfig():
    def __init__(self,
                 model = None,
                 optimizer = None,
                 criterion = None,
                 scheduler = None,
                 n_epochs = 50,
                 device = 'cpu',
                 patience = 5,
                 save_name = './deepec.log',
                 train_source = None,
                 val_source = None, 
                 test_source = None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.n_epochs = num_epochs
        self.device = device
        self.patience = patience
        self.save_name = save_name
        self.train_source = trainDataloader
        self.val_source = validDataloader
        self.test_source = testDataloader


if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    output_dir = options.output_dir
    log_dir = options.log_dir

    device = options.gpu
    num_epochs = options.epoch
    batch_size = options.batch_size
    learning_rate = options.learning_rate
    patience = options.patience

    checkpt_file = options.checkpoint
    input_data_file = options.seq_file

    third_level = options.third_level
    num_cpu = options.cpu_num

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f'{output_dir}/{log_dir}')
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    seed_num = 123 # random seed for reproducibility
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)

    torch.set_num_threads(num_cpu)

    gamma = 3

    logging.info(f'\nInitial Setting\
                  \nEpoch: {num_epochs}\
                  \tGamma: {gamma}\
                  \tBatch size: {batch_size}\
                  \tLearning rate: {learning_rate}\
                  \tPredict upto 3 level: {third_level}')


    input_seqs, input_ecs, input_ids = read_EC_Fasta(input_data_file)

    train_seqs, test_seqs = train_test_split(input_seqs, test_size=0.1, random_state=seed_num)
    train_ecs, test_ecs = train_test_split(input_ecs, test_size=0.1, random_state=seed_num)
    # train_ids, test_ids = train_test_split(input_ids, test_size=0.1, random_state=seed_num)

    train_seqs, val_seqs = train_test_split(train_seqs, test_size=1/9, random_state=seed_num)
    train_ecs, val_ecs = train_test_split(train_ecs, test_size=1/9, random_state=seed_num)
    # train_ids, val_ids = train_test_split(input_ids, test_size=1/9, random_state=seed_num)


    logging.info(f'Number of sequences used- Train: {len(train_seqs)}')
    logging.info(f'Number of sequences used- Validation: {len(val_seqs)}')
    logging.info(f'Number of sequences used- Test: {len(test_seqs)}')

    explainECs = []
    for ecs in input_ecs:
        explainECs += ecs
    explainECs = list(set(explainECs))
    explainECs.sort()

    if third_level:
        logging.info('Predict EC number upto third level')
        explainECs = getExplainedEC_short(explainECs)
        train_ecs = convertECtoLevel3(train_ecs)
        val_ecs = convertECtoLevel3(val_ecs)
        test_ecs = convertECtoLevel3(test_ecs)
    else:
        logging.info('Predict EC number upto fourth level')

    train_ec_types = []
    for ecs in train_ecs:
        train_ec_types += ecs
    len_train_ecs = len(set(train_ec_types))

    val_ec_types = []
    for ecs in val_ecs:
        val_ec_types += ecs
    len_val_ecs = len(set(val_ec_types))
    
    test_ec_types = []
    for ecs in test_ecs:
        test_ec_types += ecs
    len_test_ecs = len(set(test_ec_types))

    logging.info(f'Number of ECs in train data: {len_train_ecs}')
    logging.info(f'Number of ECs in validation data: {len_val_ecs}')
    logging.info(f'Number of ECs in test data: {len_test_ecs}')

    trainDataset = ECDataset(train_seqs, train_ecs, explainECs)
    valDataset = ECDataset(val_seqs, val_ecs, explainECs)
    testDataset = ECDataset(test_seqs, test_ecs, explainECs)
    trainDataset_emb = ECEmbedDataset(train_seqs, train_ecs, explainECs)
    valDataset_emb = ECEmbedDataset(val_seqs, val_ecs, explainECs)
    testDataset_emb = ECEmbedDataset(test_seqs, test_ecs, explainECs)

    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    validDataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)
    trainDataloader_emb = DataLoader(trainDataset_emb, batch_size=batch_size, shuffle=True)
    validDataloader_emb = DataLoader(valDataset_emb, batch_size=batch_size, shuffle=True)
    testDataloader_emb = DataLoader(testDataset_emb, batch_size=batch_size, shuffle=False)

    ntokens = 21
    emsize = 64 # embedding dimension
    nhid = 128 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = HybridModel(ntokens, emsize, nhead, nhid, nlayers, dropout, explainECs).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1])
    logging.info(f'Model Architecture: \n{model}')
    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_train_params}')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, )
    criterion = FocalLoss(gamma=gamma)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    logging.info(f'Learning rate scheduling: step size: 1\tgamma: 0.95')

    config = DeepECConfig()
    config.model = model 
    config.optimizer = optimizer
    config.criterion = criterion
    config.scheduler = scheduler
    config.n_epochs = num_epochs
    config.device = device
    config.save_name = f'{output_dir}/{checkpt_file}'
    config.patience = patience
    config.train_source = trainDataloader
    config.val_source = validDataloader
    config.test_source = testDataloader
    config.train_source_emb = trainDataloader_emb
    config.val_source_emb = validDataloader_emb
    config.test_source_emb = testDataloader_emb
    config.explainProts = explainECs


    avg_train_losses, avg_val_losses = train_emb(config)
    save_losses(avg_train_losses, avg_val_losses, output_dir=output_dir)
    draw(avg_train_losses, avg_val_losses, output_dir=output_dir)

    ckpt = torch.load(f'{output_dir}/{checkpt_file}')
    model.load_state_dict(ckpt['model'])

    y_true, y_score, y_pred = evalulate_emb(config)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    logging.info(f'(Macro) Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    logging.info(f'(Micro) Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    
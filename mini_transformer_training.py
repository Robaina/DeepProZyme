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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from deepec.process_data import read_EC_Fasta, \
                                getExplainedEC_short, \
                                convertECtoLevel3
from deepec.data_loader import ECEmbedDataset
from deepec.utils import argument_parser, draw, save_losses, FocalLoss, DeepECConfig
from deepec.train import train_mask, evalulate_mask
from deepec.model import DeepTransformer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')

class WarmupOpt:
    def __init__(self, optimizer, model_size, warmup_step):
        self.optimizer = optimizer
        self.model_size = model_size
        self.warmup_step = warmup_step
        self._step = 0
        self._rate = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.model_size**(-0.5)*min((step/100)**(-0.5), (step/100)*self.warmup_step**(-1.5))
    
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step+1 < num_warmup_steps:
            return float(current_step+1) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps+1 - current_step-1) / float(max(1, num_training_steps+1 - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


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

    gamma = 0

    logging.info(f'\nInitial Setting\
                  \nEpoch: {num_epochs}\
                  \tGamma: {gamma}\
                  \tBatch size: {batch_size}\
                  \tLearning rate: {learning_rate}\
                  \tGPU: {device}\
                  \tPredict upto 3 level: {third_level}')
    logging.info(f'Input file directory: {input_data_file}')


    input_seqs, input_ecs, _ = read_EC_Fasta(input_data_file)
    data_num = 2000000
    rand_ind = np.arange(len(input_ecs))
    random.shuffle(rand_ind)
    rand_ind = rand_ind[:data_num]
    input_seqs = [input_seqs[ind] for ind in rand_ind]
    input_ecs = [input_ecs[ind] for ind in rand_ind]
    

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

    trainDataset = ECEmbedDataset(train_seqs, train_ecs, explainECs)
    valDataset = ECEmbedDataset(val_seqs, val_ecs, explainECs)
    testDataset = ECEmbedDataset(test_seqs, test_ecs, explainECs)

    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    validDataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)

    ntokens = 21
    emsize = 64 # embedding dimension
    nhid = 128 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    logging.info(f'Network architecture info\n\
                    ntoken {ntokens}\temsize {emsize}\tnhid {nhid}\tnlayers {nlayers}\tnhead {nhead}')
    model = DeepTransformer(ntokens, emsize, nhead, nhid, nlayers, dropout, explainECs)
    # model = DeepTransformer_linear(ntokens, emsize, nhead, nhid, nlayers, dropout, explainECs)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)
    logging.info(f'Model Architecture: \n{model}')
    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_train_params}')

    optimizer_adam = optim.Adam(model.parameters(), lr=learning_rate, )
    optimizer = WarmupOpt(optimizer_adam, emsize, warmup_step=800)
    scheduler = None
    logging.info(f'Learning rate scheduling: Warmup, step: 800')

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, )
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    # logging.info(f'Learning rate scheduling: step size: 1\tgamma: 0.95')
    # warmup_step = 3
    # scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, num_epochs)
    # logging.info(f'Learning rate scheduling: linear warmup to {warmup_step}\tLambdaLR')

    criterion = FocalLoss(gamma=gamma)


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
    config.explainProts = explainECs


    avg_train_losses, avg_val_losses = train_mask(config)
    save_losses(avg_train_losses, avg_val_losses, output_dir=output_dir)
    draw(avg_train_losses, avg_val_losses, output_dir=output_dir)

    ckpt = torch.load(f'{output_dir}/{checkpt_file}')
    model.load_state_dict(ckpt['model'])

    y_true, y_score, y_pred = evalulate_mask(config)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    logging.info(f'(Macro) Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    logging.info(f'(Micro) Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    
    # len_ECs = len(explainECs)

    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # prec = dict()
    # rec = dict()
    # f1s = dict()

    # for i in range(len_ECs):
    #     fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #     prec[i] = precision_score(y_true[:, i], y_pred[:, i], )
    #     rec[i] = recall_score(y_true[:, i], y_pred[:, i])
    #     f1s[i] = f1_score(y_true[:, i], y_pred[:, i])

    # fp = open(f'{output_dir}/performance_indices.txt', 'w')
    # fp.write('EC\tAUC\tPrecision\tRecall\tF1\n')
    # for ind in roc_auc:
    #     ec = explainECs[ind]
    #     fp.write(f'{ec}\t{roc_auc[ind]}\t{prec[ind]}\t{rec[ind]}\t{f1s[ind]}\n')
    # fp.close()
# -*- encoding: utf-8 -*
# here put the import lib
import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from generators.data import SeqDataset, CL4SRecDataset, Seq2SeqDataset, Seq2SeqCL4SRecDataset
from utils.utils import unzip_data, concat_data


class Generator(object):

    def __init__(self, args, logger, device):

        self.args = args
        self.aug_file = args.aug_file
        self.inter_file = args.inter_file
        self.dataset = args.dataset
        self.num_workers = args.num_workers
        self.bs = args.train_batch_size
        self.logger = logger
        self.device = device
        self.aug_seq = args.aug_seq

        self.logger.info("Loading dataset ... ")
        start = time.time()
        self._load_dataset()
        end = time.time()
        self.logger.info("Dataset is loaded: consume %.3f s" % (end - start))

    
    def _load_dataset(self):
        '''Load train, validation, test dataset'''

        usernum = 0
        itemnum = 0
        User = defaultdict(list)    # default value is a blank list
        user_train = {}
        user_valid = {}
        user_test = {}
        # assume user/item index starting from 1
        if self.aug_seq:
            f = open('./data/%s/aug/%s.txt' % (self.dataset, self.aug_file), 'r')
        else:
            f = open('./data/%s/handled/%s.txt' % (self.dataset, self.inter_file), 'r')
        for line in f:  # use a dict to save all seqeuces of each user
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
        
        self.user_num = usernum
        self.item_num = itemnum

        for user in tqdm(User):
            nfeedback = len(User[user]) - self.args.aug_seq_len
            #nfeedback = len(User[user])
            if nfeedback < 3:
            #if nfeedback < 5:
                user_train[user] = User[user] # only for train
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2] # for train
                #user_train[user] = User[user][:-4]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                #user_valid[user].append(User[user][-4])
                user_test[user] = []
                user_test[user].append(User[user][-1])
                #user_test[user].append(User[user][-3])
        
        self.train = user_train
        self.valid = user_valid
        self.test = user_test


    
    def make_trainloader(self):

        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        train_dataset = SeqDataset(train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
    

        return train_dataloader


    def make_evalloader(self, test=False):

        if test:
            eval_dataset = concat_data([self.train, self.valid, self.test])

        else:
            eval_dataset = concat_data([self.train, self.valid])

        eval_dataset = SeqDataset(eval_dataset, self.item_num, self.args.max_len, self.args.test_neg)
        eval_dataloader = DataLoader(eval_dataset,
                                    sampler=SequentialSampler(eval_dataset),
                                    batch_size=100,
                                    num_workers=self.num_workers)
        
        return eval_dataloader

    
    def get_user_item_num(self):

        return self.user_num, self.item_num



class CL4SRecGenerator(Generator):

    def __init__(self, args, logger, device):
        
        super().__init__(args, logger, device)

    
    def make_trainloader(self):

        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        train_dataset = CL4SRecDataset(self.args, train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
    

        return train_dataloader
    

class Seq2SeqGenerator(Generator):

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)
    

    def make_trainloader(self):

        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        train_dataset = Seq2SeqDataset(self.args, train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader



class Seq2SeqCL4SRecGenerator(Generator):

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)
    

    def make_trainloader(self):

        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        train_dataset = Seq2SeqCL4SRecDataset(self.args, train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader
    
    

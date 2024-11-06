# -*- encoding: utf-8 -*-
# here put the import lib
import copy
import numpy as np
from generators.generator import Generator
from utils.utils import unzip_data, filter_data, concat_data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler



class DiffusionGenerator(Generator):

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)
        self.aug_file = args.aug_file
        self.aug_num = args.aug_num
        self.max_len = args.max_len


    def make_trainloader(self):

        train_dataset = unzip_data(self.train, aug=self.args.aug)
        train_dataset = filter_data(train_dataset, thershold=self.aug_num+2) # 过滤数量小于5的user数据。
        train_dataset = DiffusionTrainDataset(train_dataset, self.item_num, self.aug_num, self.max_len) # 对dataset进行重构，这是个不错的写法

        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers,
                                      drop_last=True)

        return train_dataloader
    

    def make_evalloader(self, test=False):

        if test:
            eval_dataset = concat_data([self.train, self.valid, self.test])

        else:
            eval_dataset = concat_data([self.train, self.valid])
        eval_dataset = filter_data(eval_dataset, thershold=self.aug_num+2)

        eval_dataset = DiffusionTrainDataset(eval_dataset, self.item_num, self.args.aug_num, self.args.max_len)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=SequentialSampler(eval_dataset),
                                     batch_size=100,
                                     num_workers=self.num_workers)
        
        return eval_dataloader
    

    def make_augmentloader(self):
        
        aug_dataset = concat_data([self.train, self.valid])
        #aug_dataset = unzip_data(self.train, aug=False)
        aug_dataset = DiffusionAugmentDataset(aug_dataset, self.item_num, self.aug_num, self.args.max_len)

        aug_dataloader = DataLoader(aug_dataset,
                                    sampler=SequentialSampler(aug_dataset),
                                    batch_size=100,
                                    num_workers=self.num_workers)

        return aug_dataloader

    
    def save_aug(self, aug_data):

        aug_data = aug_data.tolist()

        res_data = []

        for i in range(len(aug_data)):

            per_data = aug_data[i] + self.train[i+1] + self.valid[i+1] + self.test[i+1]
            res_data.append(per_data)
        
        with open('./data/%s/aug/%s.txt' % (self.dataset, self.aug_file), 'w') as f:
        
            for user in range(len(aug_data)):

                for item in res_data[user]:

                    f.write('%s %s\n' % (int(user+1), int(item)))



class DiffusionAugGenerator(DiffusionGenerator):

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)


    def save_aug(self, aug_data):

        aug_data = aug_data.tolist()

        res_data = []

        for i, user in enumerate(self.train.keys()):

            per_data = aug_data[i] + self.train[user] + self.valid[user] + self.test[user]
            res_data.append(per_data)
        
        with open('./data/%s/aug/%s.txt' % (self.dataset, self.aug_file), 'w') as f:
        
            for i, user in enumerate(self.train.keys()):

                for item in res_data[i]:

                    f.write('%s %s\n' % (int(user), int(item)))




class DiffusionTrainDataset(Dataset):

    def __init__(self, data, item_num, seq_len, max_len) -> None:

        super().__init__()
        self.data = data
        self.item_num = item_num
        self.seq_len = seq_len
        self.max_len = max_len
    
    
    def __len__(self):

        return len(self.data)
    

    def __getitem__(self, index):
        
        inter = self.data[index][self.seq_len:] # 这个是后面的序列
        diff_seq = copy.deepcopy(self.data[index][:self.seq_len])
        #diff_seq.reverse()  # whether reverse the generated sequences
        diff_seq = np.array(diff_seq) # 正常的seq，【1，3，5】一样的长度，前3个比如

        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter): # ti get the reversed sequence[0,0,0,1,3,5],just reverse
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - len(inter)
            positions = list(range(1, len(inter)+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions) # 这不是新的generator

        return seq, positions, diff_seq



class DiffusionAugmentDataset(Dataset):

    def __init__(self, data, item_num, seq_len, max_len) -> None:

        super().__init__()
        self.data = data
        self.item_num = item_num
        self.seq_len = seq_len
        self.max_len = max_len
    
    
    def __len__(self):

        return len(self.data)
    

    def __getitem__(self, index): # 在aug的时候是直接生成的。所以不需要区分diff_seq和seq
        
        inter = self.data[index]

        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in inter: # ti get the reversed sequence
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - len(inter)
            positions = list(range(1, len(inter)+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions




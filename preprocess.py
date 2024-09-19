import time
import math
import random
import numpy as np
import pandas as pd
from tqdm import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

def split_targets(targets):
    cores = cpu_count()
    part = len(targets)//cores
    
    parts = []
    for i in range(cores-1):
        temp = targets[i*part: (i+1)*part]
        parts.append(temp)
    parts.append(targets[(i+1)*part:])
    return parts, cores
    
# Gather Expression dataset
def GenData(paras):  # (gene_part, sub_tf, sub_tf_exp, genesexp_part, sub_gold, time_lag)
    error = 0
    all_tf, all_target = [], []
    gene_pair, exp_pair, labels = [], [], []
    tfs, targets, exp, gold, core = paras
    
    if core == 1:
        for i in trange(len(tfs)):
            tf = tfs[i]
            tf_exp = exp[tf]
            for target in targets:
                target_exp = exp[target]
                relation = [tf, target]
                gene_pair.append(relation)
                exp_pair.append(np.vstack((tf_exp, target_exp)))
                if relation in gold:
                    labels.append(1)
                else:
                    labels.append(0)
    else:
        for tf in tfs:
            tf_exp = exp[tf]
            for target in targets:
                target_exp = exp[target]
                relation = [tf, target]
                gene_pair.append(relation)
                exp_pair.append(np.vstack((tf_exp, target_exp)))
                if relation in gold:
                    labels.append(1)
                else:
                    labels.append(0)
    return (gene_pair, exp_pair, labels)

def processing(tf_raw, gene_raw, exp, gold):
    print('Multi-core processing...')
    start = time.time()
    if __name__ == '__main__':
        tf = np.intersect1d(tf_raw, gold[:,0].reshape(-1,))
        targets = np.intersect1d(gene_raw, gold[:,1].reshape(-1,))
        print('num of tfs:', len(tf), 'num of targets:', len(targets))
        
        # targets_part, cores = split_targets(targets) 
        # gene_pair, all_exp, all_labels = [], [], []
        # p = ProcessPoolExecutor(max_workers=cores)
        # process = [p.submit(GenData, (tf, targets_part[i], exp, gold.tolist(), i)) for i in range(cores)]
        # p.shutdown()
        # for j in range(cores):
        #     results = process[j].result()
        #     gene_pair.extend(results[0])
        #     all_exp.extend(results[1])
        #     all_labels.extend(results[2])

        gene_pair, all_exp, all_labels = GenData((tf, targets, exp, gold.tolist(), 1))

        end = time.time()
        RuningTime = end - start
        print('multiprocessing Done! Runing Time:', round(RuningTime / 60, 2), 'sec')

    return gene_pair, all_exp, all_labels

def split_datasets(labels):
    print('labels', np.sum(labels))
    pos_index, neg_index = [], []
    pos_index = [index for index, value in enumerate(labels) if value == 1]
    neg_index = [index for index, value in enumerate(labels) if value == 0]
    pos_shuffle, neg_shuffle = random.sample(pos_index, len(pos_index)), random.sample(neg_index, len(neg_index))
    pos_part, neg_part = len(pos_shuffle) // 5, len(neg_shuffle) // 5
    pos_train, neg_train = pos_shuffle[ :3*pos_part], neg_shuffle[ :3*neg_part]
    pos_val, neg_val = pos_shuffle[3*pos_part : 4*pos_part], neg_shuffle[3*neg_part : 4*neg_part]
    pos_test, neg_test = pos_shuffle[4*pos_part: ], neg_shuffle[4*neg_part: ]
    train_index = pos_train + neg_train
    val_index = pos_val + neg_val
    test_index = pos_test + neg_test

    return train_index, val_index, test_index



# Gather Expression dataset
class Feeder(Dataset):
    def __init__(self, exp_data, label, patch_size, mode='pretrain', base=128):
        assert mode=='pretrain' or mode=='lincls', 'mode should in [pretrain, lincls]'

        self.exp_data = exp_data[:, :, :exp_data.shape[-1]-exp_data.shape[-1] % patch_size]
        self.label = label
        self.arange = np.arange(exp_data.shape[-1])

    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, index):
        
        tf, target = copy.deepcopy(self.exp_data[index][0]), copy.deepcopy(self.exp_data[index][1])
        label = self.label[index]
        
        X = np.concatenate([tf.reshape(1,-1), target.reshape(1,-1)], axis=0).astype(np.float16)
        return X, label

if __name__ == '__main__':
    batch_size = 512
    patch_size = 8
    
    ExpressionData = pd.read_csv('../Benchmark Dataset/Non-Specific Dataset/hESC/TFs+1000/BL--ExpressionData.csv', index_col=0, engine='c')
    network = pd.read_csv('../Benchmark Dataset/Non-Specific Dataset/hESC/TFs+1000/Label.csv', index_col=0, engine='c').values
    tfs_raw = pd.read_csv('../Benchmark Dataset/Non-Specific Dataset/hESC/TFs+1000/TF.csv', index_col=0, engine='c')['index'].values.tolist()
    targets_raw = pd.read_csv('../Benchmark Dataset/Non-Specific Dataset/hESC/TFs+1000/Target.csv', index_col=0, engine='c')['index'].values.tolist()
    
    gene_pair, all_exp, labels = processing(tfs_raw, targets_raw, ExpressionData.values, network)
    
    data = Feeder(np.array(all_exp), labels, patch_size, 'pretrain')
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    labels_unique, counts = np.unique(labels, return_counts=True)
    class_weight = [sum(counts)/i for i in counts]
    print('labels', np.sum(labels), 'postive VS negative:', class_weight, 'Density:', round(class_weight[0]/sum(class_weight), 3))
import torch, random, os, math 
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torch_geometric.data import DataLoader
from TFM.Dataset import MolNet
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def load_data(dataset, batch_size, valid_size, test_size, cpus_per_gpu, task, split, seed=426, save_splits=False):
    data = MolNet(root='./dataset', dataset=dataset)

    if split == 'balan_scaffold':
        trainset, validset, testset = balanscaffold_split(data, valid_size, test_size)
    else:
        if split == 'random_scaffold':
            scaffold = True
        elif split == 'random':
            scaffold = False
        else:
            raise ValueError('Invalid split type')
        cont = True
        while cont:
            trainset, validset, testset = randomscaffold_split(data, valid_size, test_size, scaffold=scaffold, seed=seed)
            if task == 'clas':
                vy = [d.y for d in validset]; ty = [d.y for d in testset]
                vy = torch.cat(vy, 0); ty = torch.cat(ty, 0)
                if torch.any(torch.mean(vy, 0) == 1) or torch.any(torch.mean(vy, 0) == 0) or torch.any(torch.mean(ty, 0) == 1) or torch.any(torch.mean(ty, 0) == 0):
                    cont = True
                    if seed is not None:
                        seed += 10
                else:
                    cont = False
            else:
                cont = False
    # 如果需要保存划分，则将数据集保存为CSV
    if save_splits:
        save_dataset_splits(trainset, validset, testset, dataset, split, seed)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)
    return train_loader, valid_loader, test_loader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

class metrics_c(nn.Module):
    def __init__(self, acc_f, pre_f, rec_f, f1_f, auc_f):
        super(metrics_c, self).__init__()
        self.acc_f = acc_f
        self.pre_f = pre_f
        self.rec_f = rec_f
        self.f1_f = f1_f
        self.auc_f = auc_f

    def forward(self, out, prob, tar):
        if len(out.shape) > 1:
            acc, f1, pre, rec, auc = [], [], [], [], []
            
            for i in range(out.shape[-1]):
                acc_, f1_, pre_, rec_, auc_ = 0, 0, 0, 0, 0
                acc_ = self.acc_f(tar[:, i], out[:, i])
                f1_ = self.f1_f(tar[:, i], out[:, i])
                pre_ = self.pre_f(tar[:, i], out[:, i])
                rec_ = self.rec_f(tar[:, i], out[:, i])
                try:
                    auc_ = self.auc_f(tar[:, i], prob[:, i])
                    auc.append(auc_)
                except:pass
                
                acc.append(acc_); f1.append(f1_); pre.append(pre_); rec.append(rec_)
            return np.mean(acc), np.mean(f1), np.mean(pre), np.mean(rec), np.mean(auc)
        else:
            acc = self.acc_f(tar, out)
            f1 = self.f1_f(tar, out)
            pre = self.pre_f(tar, out)
            rec = self.rec_f(tar, out)
            auc = self.auc_f(tar, prob)
            return acc, f1, pre, rec, auc


class metrics_r(nn.Module):
    def __init__(self, mae_f, rmse_f, r2_f):
        super(metrics_r, self).__init__()
        self.mae_f = mae_f
        self.rmse_f = rmse_f
        self.r2_f = r2_f

    def forward(self, out, tar):
        mae, rmse, r2 = 0, 0, 0
        if self.mae_f is not None:
            mae = self.mae_f(tar, out)

        if self.rmse_f is not None:
            rmse = self.rmse_f(tar, out, squared=False)

        if self.r2_f is not None:
            r2 = self.r2_f(tar, out)

        return mae, rmse, r2, None, None


def create_ffn(task, tasks, output_dim, dropout):
    if task == 'clas':
        act = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim*2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(output_dim, tasks),
            nn.Sigmoid())
    elif task == 'reg':
        act = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(output_dim, tasks))
    else:
        raise NameError('task must be reg or clas!')
    return act


def get_attn_pad_mask(mask): 
    batch_size, len_q = mask.size(0), mask.size(1)
    a = mask.unsqueeze(1).expand(batch_size, len_q, len_q)
    pad_attn_mask = a * a.transpose(-1, -2) 
    return pad_attn_mask.data.eq(0) 


def balanscaffold_split(data, validrate, testrate):
    trainrate = 1 - validrate - testrate
    assert trainrate > 0.4

    train_inds, valid_inds, test_inds = [], [], []
    scaffolds = {}
    for ind, dat in enumerate(data):
        mol = Chem.MolFromSmiles(dat.smi)
        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in sorted(scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]  

    n_total_valid = round(validrate * len(data))
    n_total_test = round(testrate * len(data))
    for scaffold_set in scaffold_sets:
        if (len(valid_inds) + len(scaffold_set) <= n_total_valid) and (len(scaffold_set) < n_total_valid*0.5):
            valid_inds.extend(scaffold_set)
        elif (len(test_inds) + len(scaffold_set) <= n_total_test) and (len(scaffold_set) < n_total_test*0.5):
            test_inds.extend(scaffold_set)
        else:
            train_inds.extend(scaffold_set)
    return data[train_inds], data[valid_inds], data[test_inds]


def randomscaffold_split(data, validrate, testrate, scaffold=True, seed=426):
    trainrate = 1 - validrate - testrate
    assert trainrate > 0.4
    lenth = len(data)
    g1 = int(lenth*trainrate)
    g2 = int(lenth*(trainrate+validrate))
    
    if not scaffold:
        rng = np.random.RandomState(seed)
        random_num = list(range(lenth))
        rng.shuffle(random_num)
        data = data[random_num]
        return data[:g1], data[g1:g2], data[g2:]

    else:
        train_inds, valid_inds, test_inds = [], [], []
        scaffolds = {}
        for ind, dat in enumerate(data):
            mol = Chem.MolFromSmiles(dat.smi)
            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=True)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)
        
        rng = np.random.RandomState(seed)
        scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

        n_total_valid = round(validrate * len(data))
        n_total_test = round(testrate * len(data))
        for scaffold_set in scaffold_sets:
            if len(valid_inds) + len(scaffold_set) <= n_total_valid:
                valid_inds.extend(scaffold_set)
            elif len(test_inds) + len(scaffold_set) <= n_total_test:
                test_inds.extend(scaffold_set)
            else:
                train_inds.extend(scaffold_set)

        return data[train_inds], data[valid_inds], data[test_inds]


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR}
    formatter = logging.Formatter("[%(asctime)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger



def save_dataset_splits(trainset, validset, testset, dataset_name, split_type, seed=None, output_dir='./dataset/splits'):
    """
    将划分后的数据集保存为CSV文件
    
    Args:
        trainset, validset, testset: 划分后的数据集
        dataset_name: 数据集名称
        split_type: 划分方式
        seed: 随机种子
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    seed_info = f"_seed{seed}" if seed is not None else ""
    
    # 处理并保存每个数据集
    for name, dataset in [('train', trainset), ('valid', validset), ('test', testset)]:
        # 从数据集中提取SMILES和标签
        smiles = []
        labels = []
        
        for data in dataset:
            smiles.append(data.smi)
            labels.append(data.y.cpu().numpy())
        
        # 创建DataFrame
        df = pd.DataFrame({'smiles': smiles})
        
        # 处理标签(支持多任务和单任务)
        if len(labels[0].shape) > 0 and labels[0].shape[0] > 1:
            for i in range(labels[0].shape[0]):
                df[f'task_{i+1}'] = [label[i] for label in labels]
        else:
            df['target'] = [label[0] if hasattr(label, '__len__') else label for label in labels]
        
        # 保存为CSV
        output_path = os.path.join(output_dir, f'{dataset_name}_{name}_{split_type}{seed_info}.csv')
        df.to_csv(output_path, index=False)
        print(f"保存{name}数据集({len(df)}条记录)到: {output_path}")
    
    return output_dir
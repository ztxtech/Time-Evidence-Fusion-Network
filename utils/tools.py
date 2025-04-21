import argparse
import json
import math
import random
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def save_args_to_json(args, filename):
    """
    将命令行参数对象保存为JSON文件

    参数:
    args: 命令行参数对象（通常是argparse的Namespace或dict）
    filename: 输出的JSON文件路径
    """
    # 将参数对象转换为字典
    if isinstance(args, dict):
        arg_dict = args
    else:
        arg_dict = vars(args)

    # 保存为JSON
    with open(filename, 'w') as f:
        json.dump(arg_dict, f, indent=4, sort_keys=True)


def get_setting(args):
    setting = '{}_{}_p{}_e{}_N{}_T{}_C{}_{}_R{}_P{}_D{}_{}'.format(
        args.model,
        args.data,
        args.pred_len,
        args.e_layers,
        int(args.use_norm),
        int(args.use_T_model),
        int(args.use_C_model),
        args.fusion_method,
        int(args.use_residual),
        int(args.use_probabilistic_layer),
        args.dropout,
        uuid.uuid4().hex[:6]
    )

    return setting


def load_config(config_path):
    with open(config_path, 'r') as f:
        args = f.read()
    args = argparse.Namespace(**json.loads(args))
    return args


def dict_eq(dict1, dict2):
    common_keys = set(dict1.keys()) & set(dict2.keys())
    for key in common_keys:
        if dict1[key] != dict2[key]:
            return False
    return True


def get_all_json_paths(directory_path: str, recursive: bool = False) -> list[str]:
    path = Path(directory_path)
    if not path.is_dir():
        raise NotADirectoryError(f"{directory_path} 不是有效目录")

    # 根据 recursive 参数选择搜索方式
    search = path.rglob("**/*.json") if recursive else path.glob("*.json")

    return [
        str(file.resolve())
        for file in search
        if file.is_file()
    ]


def set_random_seed(seed=2021):
    """封装随机种子设置函数，确保所有随机数生成器的确定性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 补充CUDA种子设置
    torch.backends.cudnn.deterministic = True  # 补充cuDNN确定性设置
    torch.backends.cudnn.benchmark = False

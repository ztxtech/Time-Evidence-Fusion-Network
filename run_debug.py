import argparse
import json
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args


def get_setting(args, ii):
    setting = '{}_{}_e{}_N{}_T{}_C{}_{}_R{}_P{}_D{}'.format(
        args.model,
        args.data,
        args.e_layers,
        int(args.use_norm),
        int(args.use_T_model),
        int(args.use_C_model),
        args.fusion_method,
        int(args.use_residual),
        int(args.use_probabilistic_layer),
        args.dropout
    )

    return setting


def load_config(config_path):
    with open(config_path, 'r') as f:
        args = f.read()
    args = argparse.Namespace(**json.loads(args))
    return args


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # config_path = "{your chosen config file path}"
    config_path = "configs/comparision/ETT_script/TEFN_ETTh1_p96.json"

    args = load_config(config_path)

    args.use_gpu = True \
        if (torch.cuda.is_available()
            or torch.backends.mps.is_available()) \
        else False

    print(args.use_gpu)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    ##### Debug Hyperparameters Segment Head
    args.kernel_activation = 'mish'
    args.learning_rate = 1e-3

    #### Debug Hyperparameters Segment Tail

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        exit()

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = get_setting(args, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = get_setting(args, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

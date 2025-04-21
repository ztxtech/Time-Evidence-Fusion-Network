import random
from argparse import Namespace
from time import time

import numpy as np
import pandas as pd
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import load_config

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    config_path = 'configs/ablation/Traffic_script/TEFN_p720_fusion_method_attn.json'
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

    data = []
    for num_workers in range(2, 16, 2):
        for factor in range(32, 200, 32):
            cur_args = Namespace(**vars(args))
            cur_args.num_workers = num_workers
            cur_args.prefetch_factor = factor
            if args.task_name == 'long_term_forecast':
                Exp = Exp_Long_Term_Forecast
            else:
                exit()

            if args.is_training:
                for ii in range(args.itr):
                    exp = Exp(cur_args)
                    train_data, train_loader = exp._get_data(flag='train')
                    start = time()
                    for epoch in range(1, 3):
                        for i, data in enumerate(train_loader, 0):
                            pass
                    end = time()
                    print("Finish with:{} second, num_workers={}, factor={}".format(end - start, num_workers, factor))
                    data.append([num_workers, factor, end - start])

    data = pd.DataFrame(data, columns=['num_workers', 'factor', 'time'])
    data.to_csv('best_loader.csv', index=False)

import torch
from joblib import Parallel, delayed

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
from utils.tools import load_config, get_setting, set_random_seed, get_all_json_paths


def exp(config_path):
    set_random_seed(2021)
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
            setting = get_setting(args)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = get_setting(args)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    config_path = './configs'
    configs = get_all_json_paths(config_path, recursive=True)
    workers = 2

    Parallel(n_jobs=workers, prefer='processes')(
        delayed(exp)(config) for config in configs)

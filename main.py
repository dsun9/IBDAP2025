# coding: utf-8
import os
import sys
import json
import torch
import argparse
from utils import get_supported_methods

print('CUDA_VISIBLE_DEVICES:', os.getenv("CUDA_VISIBLE_DEVICES", 'NOT SET'))

from pytorch_lightning import seed_everything


# Parse parameters from the input
def parse_args(args):
    parser = argparse.ArgumentParser(prog='RIVGAE')
    parser.add_argument('--config', nargs=1, type=str, help='configuration file path', required=True)
    parser.add_argument('--task', type=str, default='embedding', help='task name which is needed to run', required=True)
    parser.add_argument('--method', type=str, default=None, help='graph embedding method, only used for embedding task')
    parser.add_argument('--seed', type=int, default=None, help='dataset name')
    parser.add_argument('--base_path', type=str, default=None, help='base path of the dataset')
    return parser.parse_args(args)


# Parse parameters from the json configuration file
def parse_json_args(file_path):
    config_file = open(file_path)
    json_config = json.load(config_file)
    config_file.close()
    return json_config


# Preprocessing task
def preprocessing_task(method, args):
    from preprocessing import preprocess
    assert method in ['GCRN', 'EvolveGCN']
    preprocess(method, args[method])


# Embedding task
def embedding_task(method, args):
    if method.startswith('EvolveGCN'):
        method = 'EvolveGCN'
    elif method.startswith('GCRN'):
        method = 'GCRN'
    print(args)
    assert method in get_supported_methods()

    from method.dynAE import dyngem_embedding
    from train import gnn_embedding
    args['has_cuda'] = True if torch.cuda.is_available() else False

    if not args['has_cuda'] and 'use_cuda' in args and args['use_cuda']:
        raise Exception('No CUDA devices is available, but you still try to use CUDA!')
    if 'use_cuda' in args:
        args['has_cuda'] &= args['use_cuda']
    if not args['has_cuda']:  # Use CPU
        torch.set_num_threads(args['thread_num'])

    if method in ['DynAE', 'DynRNN', 'DynAERNN']:
        dyngem_embedding(method, args)
    else:
        gnn_embedding(method, args)

def main(argv):
    args = parse_args(argv[1:])
    if args.seed is not None:
        seed_everything(args.seed, workers=True)  # Set random seed for reproducibility
    print('args:', args)
    config_dict = parse_json_args(args.config[0])
    # pass configuration parameters used in different tasks
    if args.task == 'preprocessing':
        args_dict = config_dict[args.task]
        if args.base_path is not None:
            for k in args_dict:
                args_dict[k]['base_path'] = args.base_path
        if args.seed is not None:
            for k in args_dict:
                args_dict[k]['seed'] = args.seed
        if args.method is None:
            raise AttributeError('Embedding method parameter is needed for the preprocessing task!')
        preprocessing_task(args.method, args_dict)
    elif args.task == 'embedding':
        args_dict = config_dict[args.task]
        if args.base_path is not None:
            for k in args_dict:
                args_dict[k]['base_path'] = args.base_path
        if args.seed is not None:
            for k in args_dict:
                args_dict[k]['seed'] = args.seed
        if args.method is None:
            raise AttributeError('Embedding method parameter is needed for the graph embedding task!')
        param_dict = args_dict[args.method]
        embedding_task(args.method, param_dict)
    else:
        raise AttributeError('Unsupported task!')


if __name__ == '__main__':
    main(sys.argv)

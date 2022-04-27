import os
import time
import socket
import argparse

local = False

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected boolean value.')


def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--epochs", type=int, default=2000)
    argparser.add_argument("--hidden_dim", type=int, default=32)
    argparser.add_argument("--learning_rate", type=float, default=0.01)
    argparser.add_argument("--weight_decay", type=float, default=0.01)
    argparser.add_argument("--momentum", type=float, default=0.9)
    argparser.add_argument("--amsgrad", type=str2bool, nargs='?',
                           const=True, default=False)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--activation", type=str, default='relu')
    argparser.add_argument("--early_stopping", type=str2bool, nargs='?',
                           const=True, default=True)
    argparser.add_argument("--patience", type=int, default=20)
    argparser.add_argument("--optim_type", type=str, default='sgd',
                           help='sgd or adam')
    argparser.add_argument("--parallel", type=str2bool, nargs='?',
                           const=True, default=False)
    argparser.add_argument("--seed", type=int, default=12345)
    argparser.add_argument("--subsample_rate", type=float, default=1.,
                           help='If 1. then no subsampling.')
    argparser.add_argument("--split_graph", type=str2bool, nargs='?',
                           const=True, default=False)
    argparser.add_argument("--split_n_subgraphs", type=int, default=10)

    argparser.add_argument("--private", type=str2bool, nargs='?',
                           const=True, default=False)
    argparser.add_argument("--delta", type=float, default=1e-5)
    argparser.add_argument("--gradient_norm_bound", type=float, default=1.)
    argparser.add_argument("--noise_scale", type=float, default=4)
    argparser.add_argument("--lot_size", type=int, default=1)
    argparser.add_argument("--sample_size", type=int, default=1)

    argparser.add_argument("--verbose", type=str2bool, nargs='?',
                           const=True, default=True)
    argparser.add_argument("--dataset", type=str, default='cora',
                           help='cora, citeseer, pubmed, reddit,'
                                'reddit-small, or pokec-pets')
    argparser.add_argument("--pokec_feat_type", type=str, default='bert_avg',
                           help='sbert, bert_avg, ft or bows')

    args = argparser.parse_args()

    return args


class Settings(object):
    '''
    Configuration for the project.
    '''
    def __init__(self):
        self.args = parse_arguments()

        if not self.args.private:
            self.model_name = f'E{self.args.epochs}_'\
                              f'SubSampl{self.args.subsample_rate}_'\
                              f'Hd{self.args.hidden_dim}_'\
                              f'Lr{self.args.learning_rate}_'\
                              f'Wd{self.args.weight_decay}_'\
                              f'M{self.args.momentum}_'\
                              f'D{self.args.dropout}_'\
                              f'Es{self.args.early_stopping}_'\
                              f'Pat{self.args.patience}_'\
                              f'Op{self.args.optim_type}'\
                              f'Split{self.args.split_graph}'\
                              f'Subgraphs{self.args.split_n_subgraphs}'

        else:
            self.epsilon = 0
            self.model_name = f'E{self.args.epochs}_'\
                              f'SubSampl{self.args.subsample_rate}_'\
                              f'Hd{self.args.hidden_dim}_'\
                              f'Lr{self.args.learning_rate}_'\
                              f'Wd{self.args.weight_decay}_'\
                              f'M{self.args.momentum}_'\
                              f'D{self.args.dropout}_'\
                              f'Es{self.args.early_stopping}_'\
                              f'Pat{self.args.patience}_'\
                              f'Op{self.args.optim_type}_'\
                              f'Eps{self.epsilon}_'\
                              f'Gnb{self.args.gradient_norm_bound}_'\
                              f'Ns_{self.args.noise_scale}_'\
                              f'LotS_{self.args.lot_size}_'\
                              f'SampS_{self.args.sample_size}'\
                              f'Split{self.args.split_graph}'\
                              f'Subgraphs{self.args.split_n_subgraphs}'

        if self.args.dataset == 'pokec-pets':
            self.model_name += f'PokecType_{self.args.pokec_feat_type}'

        # Setting up directory structure
        self.root_dir = 'data'
        self.out_dir = 'out'
        self.data_dir = os.path.join(self.out_dir, f'{self.args.dataset}')
        self.privacy_dir = os.path.join(self.data_dir,
                                        f'Privacy_{self.args.private}')
        self.log_dir = os.path.join(self.privacy_dir, f'log_{self.model_name}')
        self.seed_dir = os.path.join(self.log_dir, f'Seed_{self.args.seed}')
        now = time.localtime()
        self.time_dir = os.path.join(
                self.seed_dir,
                f'{now[0]}_{now[1]}_{now[2]}_{now[3]:02d}:{now[4]:02d}:{now[5]:02d}'
                )

    def make_dirs(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.privacy_dir):
            os.makedirs(self.privacy_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.seed_dir):
            os.makedirs(self.seed_dir)
        if not os.path.exists(self.time_dir):
            os.makedirs(self.time_dir)

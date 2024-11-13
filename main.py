import gc
import argparse
import torch
import random
import warnings
import numpy as np

from Engine import Engine


seed=114514
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

warnings.filterwarnings("ignore")

def get_args():
    ap = argparse.ArgumentParser(description='AdaMH hyperparameter')
    ap.add_argument(
        '--lr',
        default=0.001,
        type=float,
        help='learning rate'
    )
    ap.add_argument(
        '--weight-decay',
        default=0.001,
        type=float,
        help='weight decay'
    )
    ap.add_argument(
        '--num-ntype',
        default=6,
        type=int,
        help='num ntype'
    )
    ap.add_argument(
        '--dropout-rate',
        default=0.01,
        type=int,
        help='dropout rate'
    )
    ap.add_argument(
        '--feats-type',
        type=int,
        default=0,
        help='Type of the node features used'
    )
    ap.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='Dimension of the node hidden state. Default is 256'
    )
    ap.add_argument(
        '--num-heads',
        type=int,
        default=9,
        help='Number of the attention heads. Default is 9'
    )
    ap.add_argument(
        '--attn-vec-dim',
        type=int,
        default=64,
        help='Dimension of the attention vector. Default is 64'
    )
    ap.add_argument(
        '--rnn-type',
        default='RotatE0',
        help='Type of the aggregator. Default is RotatE0.'
    )
    ap.add_argument(
        '--epochs',
        type=int,
        default=9,
        help='Number of epochs. Default is 9'
    )
    ap.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Patience. Default is 5.'
    )
    ap.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size. Default is 64'
    )
    ap.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of neighbors sampled. Default is 100.'
    )
    ap.add_argument(
        '--repeat',
        type=int,
        default=2,
        help='Repeat the training and testing for N times. Default is 10.'
    )

    args = ap.parse_args()
    return args

def main():
    num_circrna = 3094
    num_disease = 347
    args=get_args()

    engine = Engine(num_circrna, num_disease, args)
    engine.init_model(args)
    engine.prepare_data()
    engine.run(args.repeat)

    torch.cuda.empty_cache()
    gc.collect()

if __name__=="__main__":
    main()
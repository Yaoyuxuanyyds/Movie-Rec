import os
from os.path import join
import sys
import argparse
import torch
import numpy as np
import Procedure
import utils
import model
import dataloader
from tensorboardX import SummaryWriter
import time
from pprint import pprint
import multiprocessing
from warnings import simplefilter

# ================== Argument Parsing ==================
def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the keep probability for dropout")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='ml-25m',
                        help="datasets")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="@k test list")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    return parser.parse_args()

args = parse_args()

# ================== Environment Setup ==================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
ROOT_PATH = os.getcwd()
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(ROOT_PATH, 'runs')
FILE_PATH = join(ROOT_PATH, 'checkpoints')
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)
simplefilter(action="ignore", category=FutureWarning)

sys.path.append(join(CODE_PATH, 'sources'))

# ================== Configuration ==================
config = {
    'bpr_batch_size': args.bpr_batch,
    'latent_dim_rec': args.recdim,
    'lightGCN_n_layers': args.layer,
    'dropout': args.dropout,
    'keep_prob': args.keepprob,
    'test_u_batch_size': args.testbatch,
    'lr': args.lr,
    'decay': args.decay,
}

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
if dataset in ['ml-latest-small', 'ml-25m']:
    dataset = dataloader.MoviesLoader(path="../data/"+dataset)



TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)




# ================== Initialization ==================
def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

print('===========config================')
pprint(config)
print("cores for test:", CORES)
print("LOAD:", LOAD)
print("Weight path:", PATH)
print("Test Topks:", topks)
print("using bpr loss")
print('===========end===================')


# ================== Main Training Loop ==================
def main():
    utils.set_seed(seed)
    print(">>SEED:", seed)

    Recmodel = model.LightGCN(config, dataset)
    Recmodel = Recmodel.to(device)
    bpr = utils.BPRLoss(Recmodel, config)

    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")


    w = SummaryWriter(join(BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-")))


    try:
        for epoch in range(TRAIN_epochs):
            start = time.time()
            if epoch % 10 == 0:
                cprint("[TEST]")
                Procedure.Test(dataset, Recmodel, epoch, w)
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=1, w=w)
            epoch_time = time.time() - start
            remaining_epochs = TRAIN_epochs - (epoch + 1)
            estimated_remaining_time = remaining_epochs * epoch_time

            print(f"EPOCH[{epoch+1}/{TRAIN_epochs}] {output_information};  Estimated remaining time: {estimated_remaining_time / 60:.2f} minutes")

            if epoch % 100 == 0:
                torch.save(Recmodel.state_dict(), weight_file)

    finally:
        w.close()

if __name__ == "__main__":
    main()

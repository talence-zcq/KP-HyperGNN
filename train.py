#!/usr/bin/env python
# coding: utf-8
import os
import time
# import math
import torch
# import pickle

import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F

from data.load_data import load_data
from model.load_model import parse_method
from utils.config import parse_config

from tqdm import tqdm

from utils.evaluate import count_parameters, eval_acc, evaluate
from utils.logging import Logger

# Part 0: Parse arguments
args = parse_config()
if args.cuda in [0, 1, 2, 3, 4 ,5, 6, 7]:
    device = torch.device('cuda:'+str(args.cuda)
                          if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

# # Part 1: Load data
data, split_idx_lst = load_data(args)

# # Part 2: Load model
model = parse_method(args, data)
model, data = model.to(device), data.to(device)
num_params = count_parameters(model)

# # Part 3: Main. Training + Evaluation
logger = Logger(args.runs, args)
criterion = nn.NLLLoss()
eval_func = eval_acc
model.train()
# print('MODEL:', model)

### Training loop ###
runtime_list = []
for run in tqdm(range(args.runs)):
    start_time = time.time()
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    if args.method == 'UniGCNII':
        optimizer = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=0.01),
            dict(params=model.non_reg_params, weight_decay=5e-4)
        ], lr=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val = float('-inf')
    for epoch in range(args.epochs):
        #         Training part
        model.train()
        optimizer.zero_grad()
        out = model(data)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
#         if args.method == 'HNHN':
#             scheduler.step()
#         Evaluation part
        result = evaluate(model, data, split_idx, eval_func)
        logger.add_result(run, result[:3])

        if epoch % args.display_step == 0 and args.display_step > 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Train Loss: {loss:.4f}, '
                  f'Valid Loss: {result[4]:.4f}, '
                  f'Test  Loss: {result[5]:.4f}, '
                  f'Train Acc: {100 * result[0]:.2f}%, '
                  f'Valid Acc: {100 * result[1]:.2f}%, '
                  f'Test  Acc: {100 * result[2]:.2f}%')

    end_time = time.time()
    runtime_list.append(end_time - start_time)

    # logger.print_statistics(run)

### Save results ###
avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

best_val, best_test = logger.print_statistics()
res_root = 'hyperparameter_tunning'
if not osp.isdir(res_root):
    os.makedirs(res_root)

filename = f'{res_root}/{args.dname}_noise_{args.feature_noise}.csv'
print(f"Saving results to {filename}")
with open(filename, 'a+') as write_obj:
    cur_line = f'{args.method}_{args.lr}_{args.wd}_{args.heads}'
    cur_line += f',{best_val.mean():.3f} ± {best_val.std():.3f}'
    cur_line += f',{best_test.mean():.3f} ± {best_test.std():.3f}'
    cur_line += f',{num_params}, {avg_time:.2f}s, {std_time:.2f}s'
    cur_line += f',{avg_time//60}min{(avg_time % 60):.2f}s'
    cur_line += f'\n'
    write_obj.write(cur_line)

all_args_file = f'{res_root}/all_args_{args.dname}_noise_{args.feature_noise}.csv'
with open(all_args_file, 'a+') as f:
    f.write(str(args))
    f.write('\n')

print('All done! Exit python code')
quit()

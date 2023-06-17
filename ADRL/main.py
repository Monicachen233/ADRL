#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import pickle
import time
from utils import construct_attr_matrix
from utils import Data, split_validation
from model import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='amazon_software', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ') #10
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hop', type=int, default=2, help='n_hops')
opt = parser.parse_args()
print(opt)



def main():
    if opt.dataset == 'amazon_software':
        max_item = 21664
        max_attr = 8326
        # n_node = 42771
    elif opt.dataset == 'yelp':
        max_item = 27097
        max_attr = 19
    elif opt.dataset == 'cosmetics':
        max_item = 42102
        max_attr = 37976
    else:
        max_item = 310
        max_attr = 10

    os.environ['CUDA_VISIBLE_DEVICE']='0'

    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train_attr.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test_attr.txt', 'rb'))
    kg_file = "../datasets/" + opt.dataset + "/attribute.txt"
    A_attr = construct_attr_matrix(kg_file,max_attr,max_item)
    train_data = Data(train_data, shuffle=True,A_attr=A_attr)
    test_data = Data(test_data, shuffle=False,A_attr=A_attr)

    model = trans_to_cuda(SessionGraph(opt,max_item,max_attr,A_attr))

    start = time.time()
    best_result = [0, 0, 0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0, 0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        mrr_20, hit_20, ndcg_20, hr_20, mrr_10, hit_10, ndcg_10, hr_10 = train_test(model, train_data, test_data)
        flag = 0
        if mrr_20 >= best_result[0]:
            best_result[0] = mrr_20
            best_epoch[0] = epoch
            flag = 1
        if hit_20 >= best_result[1]:
            best_result[1] = hit_20
            best_epoch[1] = epoch
            flag = 1
        if ndcg_20 >= best_result[2]:
            best_result[2] = ndcg_20
            best_epoch[2] = epoch
            flag = 1
        if hr_20 >= best_result[3]:
            best_result[3] = hr_20
            best_epoch[3] = epoch
        if mrr_10 >= best_result[4]:
            best_result[4] = mrr_10
            best_epoch[4] = epoch
            flag = 1
        if hit_10 >= best_result[5]:
            best_result[5] = hit_10
            best_epoch[5] = epoch
            flag = 1
        if ndcg_10 >= best_result[6]:
            best_result[6] = ndcg_10
            best_epoch[6] = epoch
            flag = 1
        if hr_10 >= best_result[7]:
            best_result[7] = hr_10
            best_epoch[7] = epoch
        print('Best Result:')
        print('\tMMR@20:\t%.4f\tRecall@20:\t%.4f\tNDCG@20:\t%.4f\tHR@20:\t%.4f\tMRR@10:\t%.4f\tRecall@10:\t%.4f\tNDCG@10:\t%.4f\tHR@10:\t%.4f\tepoch:\t%d,\t%d,\t%d,\t%d,\t%d,\t%d,\t%d,\t%d\t'% (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5], best_result[6], best_result[7], best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5], best_epoch[6], best_epoch[7]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()

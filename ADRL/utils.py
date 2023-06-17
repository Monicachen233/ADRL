#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import torch
import pandas as pd
# import dgl
import numpy as np

def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    # us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    # us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    us_pois = [list(reversed(upois)) + [0] * (len_max - le) if le < len_max else list(reversed(upois[-len_max:]))
               for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) if le < len_max else [1] * len_max
               for le in us_lens]
    return us_pois, us_msks, len_max

def construct_attr_matrix(attr,max_attr,max_item):
    attr_data = pd.read_csv(attr, sep='\t', names=['item','attr'], engine='python')
    attr_data = attr_data.drop_duplicates()
    A_attr = np.zeros((max_item, max_attr))
    for row in attr_data.iterrows():
        item,attr = row[1]
        A_attr[item][attr]=1
    return torch.Tensor(A_attr)


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, A_attr=None, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0]) 
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.A_attr = A_attr
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        targets = list(map(int, targets))
        items, n_node, A, alias_inputs, A_attr_sess = [], [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input))) 
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            temp = node.tolist() + (max_n_node - len(node)) * [0]
            temp = list(map(int, temp))
            items.append(temp)
            temp1 = node.tolist()
            temp1 = list(map(int,temp1))
            attr_matrix = torch.index_select(self.A_attr, 0, torch.LongTensor(temp1))
            attr_matrix =  torch.sum(attr_matrix, dim=0)
            if(attr_matrix.sum()!=0.):
                attr_matrix= torch.div(attr_matrix, attr_matrix.sum())
            attr_matrix = attr_matrix.numpy()
            A_attr_sess.append(attr_matrix)
            u_A = np.zeros((max_n_node, max_n_node))
            # for i in np.arange(len(u_input) - 1):
            #     if u_input[i + 1] == 0:
            #         break
            #     u = np.where(node == u_input[i])[0][0]
            #     v = np.where(node == u_input[i + 1])[0][0]
            #     u_A[u][v] = 1

            for i in np.arange(len(u_input) - 1):
                u = np.where(node == u_input[i])[0][0]
                u_A[u][u] = 1
                if u_input[i + 1] == 0:
                    break
                v = np.where(node == u_input[i + 1])[0][0]
                if u == v or u_A[u][v] == 4:
                    continue
                u_A[v][v] = 1
                if u_A[v][u] == 2:
                    u_A[u][v] = 4
                    u_A[v][u] = 4
                else:
                    u_A[u][v] = 2
                    u_A[v][u] = 3
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) 
    
        return alias_inputs, A, items, mask, targets, A_attr_sess

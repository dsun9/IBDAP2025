# coding: utf-8
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from embedding import BaseEmbedding
from helper import DataLoader
from metrics import ClassificationLoss
from filelock import FileLock
from models import MLPClassifier
from utils import get_roc_scores

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)



# DynAE model and its components
# Multi-linear perceptron class
class MLP(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool
    layer_list: nn.ModuleList
    layer_num: int

    def __init__(self, input_dim, output_dim, n_units, bias=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, n_units[0], bias=bias))

        layer_num = len(n_units)
        for i in range(1, layer_num):
            self.layer_list.append(nn.Linear(n_units[i - 1], n_units[i], bias=bias))
        self.layer_list.append(nn.Linear(n_units[-1], output_dim, bias=bias))
        self.layer_num = layer_num + 1

    def forward(self, x):
        for i in range(self.layer_num):
            x = F.relu(self.layer_list[i](x))
        return x


# DynAE class
class DynAE(nn.Module):
    input_dim: int
    output_dim: int
    look_back: int
    bias: bool
    method_name: str
    encoder: MLP
    decoder: MLP

    def __init__(self, input_dim, output_dim, look_back=3, n_units=None, bias=True, **kwargs):
        super(DynAE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.look_back = look_back
        self.bias = bias
        self.method_name = 'DynAE'

        self.encoder = MLP(input_dim * look_back, output_dim, n_units, bias=bias)
        self.decoder = MLP(output_dim, input_dim, n_units[::-1], bias=bias)

    def forward(self, x):
        hx = self.encoder(x)
        x_pred = self.decoder(hx)
        return hx, x_pred


# L1 and L2 regularization loss
class RegularizationLoss(nn.Module):
    nu1: float
    nu2: float

    def __init__(self, nu1, nu2):
        super(RegularizationLoss, self).__init__()
        self.nu1 = nu1
        self.nu2 = nu2

    @staticmethod
    def get_weight(model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                # print('name: ', name)
                weight_list.append(weight)
        return weight_list

    def forward(self, model):
        loss = Variable(torch.FloatTensor([0.]), requires_grad=True).cuda() if torch.cuda.is_available() else Variable(torch.FloatTensor([0.]), requires_grad=True)
        # No L1 regularization and no L2 regularization
        if self.nu1 == 0. and self.nu2 == 0.:
            return loss
        # calculate L1-regularization loss and L2-regularization loss
        weight_list = self.get_weight(model)
        weight_num = len(weight_list)
        # print('weight num', weight_num)
        l1_reg_loss, l2_reg_loss = 0, 0
        for name, weight in weight_list:
            if self.nu1 > 0:
                l1_reg = torch.norm(weight, p=1)
                l1_reg_loss = l1_reg_loss + l1_reg
            if self.nu2 > 0:
                l2_reg = torch.norm(weight, p=2)
                l2_reg_loss = l2_reg_loss + l2_reg
        l1_loss = self.nu1 * l1_reg_loss / weight_num
        l2_loss = self.nu2 * l2_reg_loss / weight_num
        return l1_loss + l2_loss


# Loss used for DynAE, DynRNN, DynAERNN
class DynGraph2VecLoss(nn.Module):
    beta: float
    regularization: RegularizationLoss

    def __init__(self, beta, nu1, nu2):
        super(DynGraph2VecLoss, self).__init__()
        self.beta = beta
        self.regularization = RegularizationLoss(nu1, nu2)

    def forward(self, model, input_list):
        x_reconstruct, x_real, y_penalty = input_list[0], input_list[1], input_list[2]
        assert len(input_list) == 3
        reconstruct_loss = torch.mean(torch.sum(torch.square((x_reconstruct - x_real) * y_penalty), dim=1))
        regularization_loss = self.regularization(model)
        # print('total loss: ', main_loss.item(), ', reconst loss: ', reconstruct_loss.item(), ', L1 loss: ', l1_loss.item(), ', L2 loss: ', l2_loss.item())
        return reconstruct_loss + regularization_loss

# Loss used for DynAE, DynRNN, DynAERNN
class DynGraph2VecClassificationLoss(nn.Module):
    beta: float
    regularization: RegularizationLoss

    def __init__(self, incl, n_class, beta, nu1, nu2):
        super(DynGraph2VecClassificationLoss, self).__init__()
        self.beta = beta
        self.regularization = RegularizationLoss(nu1, nu2)
        self.classification_loss = ClassificationLoss(n_class)
        self.incl = incl
        self.disabled = False

    def disable(self, v):
        self.disabled = v

    def forward(self, model, input_list, batch_labels):
        x_reconstruct, x_real, y_penalty, cls_list = input_list[0], input_list[1], input_list[2], input_list[3]
        assert len(input_list) == 4
        if not self.disabled:
            reconstruct_loss = torch.mean(torch.sum(torch.square((x_reconstruct - x_real) * y_penalty), dim=1))
            regularization_loss = self.regularization(model)
        else:
            reconstruct_loss = 0
            regularization_loss = 0
        if cls_list is not None:
            classification_loss, total_acc, total_auc = self.classification_loss([None, cls_list], batch_labels)
        else:
            classification_loss, total_acc, total_auc = 0, 0, 0
        # print('total loss: ', main_loss.item(), ', reconst loss: ', reconstruct_loss.item(), ', L1 loss: ', l1_loss.item(), ', L2 loss: ', l2_loss.item())
        if self.incl:
            total_loss = reconstruct_loss + regularization_loss + 10 * classification_loss
        else:
            total_loss = reconstruct_loss + regularization_loss
        return total_loss, total_acc, total_auc


# Batch generator used for DynAE, DynRNN and DynAERNN
class BatchGenerator:
    node_list: list
    node_num: int
    batch_size: int
    look_back: int
    beta: float
    shuffle: bool
    has_cuda: bool

    def __init__(self, node_list, batch_size, look_back, beta, shuffle=True, has_cuda=False):
        self.node_list = node_list
        self.node_num = len(node_list)
        self.batch_size = batch_size
        self.look_back = look_back
        self.beta = beta
        self.shuffle = shuffle
        self.has_cuda = has_cuda

    def generate(self, graph_list):
        graph_num = len(graph_list)
        train_size = graph_num - self.look_back
        assert train_size > 0
        all_node_num = self.node_num * train_size
        batch_num = all_node_num // self.batch_size
        if all_node_num % self.batch_size != 0:
            batch_num += 1
        node_indices = np.arange(all_node_num)

        if self.shuffle:
            np.random.shuffle(node_indices)
        counter = 0
        while True:
            # print('------------GENERATOR------------')
            # print(len(graph_list), graph_list[0].shape, 'num all nodes', all_node_num)
            
            batch_indices = node_indices[self.batch_size * counter: min(all_node_num, self.batch_size * (counter + 1))]
            x_pre_batch = torch.zeros((self.batch_size, self.look_back, self.node_num))
            x_pre_batch = x_pre_batch.cuda() if self.has_cuda else x_pre_batch
            x_cur_batch = torch.zeros((self.batch_size, self.node_num), device=x_pre_batch.device)
            y_batch = torch.ones(x_cur_batch.shape, device=x_pre_batch.device)  # penalty tensor for x_cur_batch

            for idx, record_id in enumerate(batch_indices):
                graph_idx = record_id // self.node_num
                node_idx = record_id % self.node_num
                # print(graph_idx, node_idx, record_id, idx)
                for step in range(self.look_back):
                    # graph is a scipy.sparse.lil_matrix
                    pre_tensor = torch.tensor(graph_list[graph_idx + step][node_idx, :].toarray(), device=x_pre_batch.device)
                    x_pre_batch[idx, step, :] = pre_tensor
                # graph is a scipy.sparse.lil_matrix
                cur_tensor = torch.tensor(graph_list[graph_idx + self.look_back][node_idx, :].toarray(), device=x_pre_batch.device)
                x_cur_batch[idx] = cur_tensor

            y_batch[x_cur_batch != 0] = self.beta
            counter += 1
            yield x_pre_batch, x_cur_batch, y_batch, batch_indices

            if counter == batch_num:
                if self.shuffle:
                    np.random.shuffle(node_indices)
                counter = 0


# Batch Predictor used for DynAE, DynRNN and DynAERNN
class BatchPredictor:
    node_list: list
    node_num: int
    batch_size: int
    has_cuda: bool

    def __init__(self, node_list, batch_size, has_cuda=False):
        self.node_list = node_list
        self.node_num = len(node_list)
        self.batch_size = batch_size
        self.has_cuda = has_cuda

    def get_predict_res(self, graph_list, model, batch_indices, counter, look_back, embedding_mat, x_pred):
        batch_size = len(batch_indices)
        x_pre_batches = torch.zeros((batch_size, look_back, self.node_num))
        x_pre_batches = x_pre_batches.cuda() if self.has_cuda else x_pre_batches
        
        for idx, node_idx in enumerate(batch_indices):
            for step in range(look_back):
                # graph is a scipy.sparse.lil_matrix
                pre_tensor = torch.tensor(graph_list[step][node_idx, :].toarray(), device=x_pre_batches.device)
                x_pre_batches[idx, step, :] = pre_tensor
        # DynAE uses 2D tensor as its input
        if model.method_name == 'DynAE':
            x_pre_batches = x_pre_batches.reshape(batch_size, -1)
        embedding_mat_batch, x_pred_batch = model(x_pre_batches)
        if counter:
            embedding_mat = torch.cat((embedding_mat, embedding_mat_batch), dim=0)
            x_pred = torch.cat((x_pred, x_pred_batch), dim=0)
        else:
            embedding_mat = embedding_mat_batch
            x_pred = x_pred_batch
        return embedding_mat, x_pred

    def predict(self, model, graph_list):
        look_back = len(graph_list)
        counter = 0
        embedding_mat, x_pred = 0, 0
        batch_num = self.node_num // self.batch_size

        while counter < batch_num:
            batch_indices = range(self.batch_size * counter, self.batch_size * (counter + 1))
            embedding_mat, x_pred = self.get_predict_res(graph_list, model, batch_indices, counter, look_back, embedding_mat, x_pred)
            counter += 1
        # has a remaining batch
        if self.node_num % self.batch_size != 0:
            remain_indices = range(self.batch_size * counter, self.node_num)
            embedding_mat, x_pred = self.get_predict_res(graph_list, model, remain_indices, counter, look_back, embedding_mat, x_pred)
        return embedding_mat, x_pred


# Dynamic Embedding for DynAE, DynRNN, DynAERNN
class DynamicEmbedding(BaseEmbedding):

    def __init__(self, base_path, origin_folder, embedding_folder, node_list, model, classifier, loss, batch_generator, batch_predictor, model_folder="model", has_cuda=False):
        super(DynamicEmbedding, self).__init__(base_path, origin_folder, embedding_folder, node_list, model, loss, model_folder=model_folder, has_cuda=has_cuda)
        self.batch_generator = batch_generator
        self.batch_predictor = batch_predictor
        self.classifier = classifier

        assert batch_generator.batch_size == batch_predictor.batch_size
        assert batch_generator.node_num == batch_predictor.node_num

    def get_batch_info(self, adj_list, model, node_labels_train, node_labels_test):
        graph_num = len(adj_list)
        batch_size = self.batch_generator.batch_size
        train_size = graph_num - self.batch_generator.look_back
        element_num = self.node_num * train_size
        batch_num = element_num // batch_size
        if element_num % batch_size != 0:
            batch_num += 1


        assert node_labels_train
        assert node_labels_test
        timestamp_num = len(node_labels_train)
        assert len(node_labels_train) == len(node_labels_test)
        device = node_labels_train[0].device

        idx_train, label_train, idx_val, label_val, idx_test, label_test = [], [], [], [], [], []
        for i in range(timestamp_num):
            cur_labels_train = node_labels_train[i]  # tensor
            cur_labels_test = node_labels_test[i]  # tensor
            assert cur_labels_train.shape[1] == 2
            assert cur_labels_test.shape[1] == 2
            train_items, train_labels = cur_labels_train[:, 0], cur_labels_train[:, 1]
            val_items, val_labels = cur_labels_test[:, 0], cur_labels_test[:, 1]
            test_items, test_labels = cur_labels_test[:, 0], cur_labels_test[:, 1]
            idx_train.append(train_items)
            label_train.append(train_labels)
            idx_val.append(val_items)
            label_val.append(val_labels)
            idx_test.append(test_items)
            label_test.append(test_labels)

        return batch_size, batch_num, train_size, idx_train, label_train, idx_val, label_val, idx_test, label_test

    def get_model_res(self, model, generator):
        batch_size = self.batch_generator.batch_size
        x_pre_batches, x_cur_batch, y_batch, ae_batch_indices = next(generator)
        # DynAE uses 2D tensor as its input
        if model.method_name == 'DynAE':
            x_pre_batches = x_pre_batches.reshape(batch_size, -1)
        _, x_pred_batch = model(x_pre_batches)
        loss_input_list = [x_pred_batch, x_cur_batch, y_batch]
        return loss_input_list

    def learn_embedding(self, is_first, adj_list, node_labels_train=None, node_labels_test=None, edge_list=None, neg_edge_list=None, epoch=50, lr=1e-3, idx=0, weight_decay=0., model_file='dynAE', classifier_file='dynAE_cls', seed=None, load_model=False, export=True):
        print('start learning embedding!')
        if seed is not None:
            model_file = model_file + '_' + str(seed)
            classifier_file = classifier_file + '_' + str(seed)
        model, loss_model, optimizer, classifier = self.prepare(load_model, model_file, classifier_file, lr=lr, weight_decay=weight_decay)
        batch_size, batch_num, train_size, idx_train, label_train, idx_val, label_val, idx_test, label_test = self.get_batch_info(adj_list, model, node_labels_train, node_labels_test)
        the_idx1, the_label1 = zip(*dict.fromkeys(zip(torch.concat(idx_train, dim=0).tolist(), torch.concat(label_train, dim=0).tolist())).keys())
        idx_train = torch.tensor(the_idx1, dtype=idx_train[-1].dtype).cuda() if self.has_cuda else torch.tensor(the_idx1, dtype=idx_train[-1].dtype)
        label_train = torch.tensor(the_label1, dtype=label_train[-1].dtype).cuda() if self.has_cuda else torch.tensor(the_label1, dtype=label_train[-1].dtype)
        
        model.train()
        print('start training!')
        st = time.time()
        for i in range(epoch):
            loss_sum = 0
            # loss_model.disable(False)
            for j in range(batch_num):
                generator = self.batch_generator.generate(adj_list)
                loss_input_list = self.get_model_res(model, generator)
                loss = loss_model(model, loss_input_list)
                loss.backward()
                loss_sum += loss.item()
                # gradient accumulation
                if j == batch_num - 1:
                    optimizer.step()  # update gradient
                    model.zero_grad()
                self.clear_cache()
                if j < batch_num:
                    print('Epoch: ' + str(i + 1), ', batch num: ' + str(j + 1), 'loss_train: {:.4f}'.format(loss.item()))
            self.clear_cache()
        print('finish training!')
        print('start predicting!')
        model.eval()
        classifier.eval()
        
        embedding_mat_last, _ = self.batch_predictor.predict(model, adj_list[train_size:])
        if is_first:
            the_idx, the_label = zip(*dict.fromkeys(zip(torch.concat(idx_test, dim=0).tolist(), torch.concat(label_test, dim=0).tolist())).keys())
            true_label = torch.tensor(the_label, dtype=label_test[-1].dtype).cuda() if self.has_cuda else torch.tensor(the_label, dtype=label_test[-1].dtype)
            cls_list_last = classifier([embedding_mat_last], [torch.tensor(the_idx, dtype=idx_test[-1].dtype).cuda() if self.has_cuda else torch.tensor(the_idx, dtype=idx_test[-1].dtype)])
        else:
            true_label = label_test[-1]
            cls_list_last = classifier([embedding_mat_last], [idx_test[-1]])
        self.clear_cache()

        print('end predicting!')
        en = time.time()
        cost_time = en - st

        if export:
            self.save_embedding(embedding_mat_last, idx)
        del adj_list, model, classifier
        self.clear_cache()
        print('learning embedding total time: ', cost_time, ' seconds!')
        return embedding_mat_last, cls_list_last[0], true_label, cost_time


def dyngem_embedding(method, args):
    assert method in ['DynAE', 'DynRNN', 'DynAERNN']
    from method.dynRNN import DynRNN
    from method.dynAERNN import DynAERNN
    model_dict = {'DynAE': DynAE, 'DynRNN': DynRNN, 'DynAERNN': DynAERNN}

    # DynAE, DynRNN, DynAERNN common params
    base_path = args['base_path']
    origin_folder = args['origin_folder']
    embedding_folder = args['embed_folder']
    model_folder = args['model_folder']
    model_file = args['model_file']
    node_file = args['node_file']
    file_sep = args['file_sep']
    start_idx = args['start_idx']
    end_idx = args['end_idx']
    duration = args['duration']
    embed_dim = args['embed_dim']
    has_cuda = args['has_cuda']
    epoch = args['epoch']
    lr = args['lr']
    batch_size = args['batch_size']
    load_model = args['load_model']
    shuffle = args['shuffle']
    export = args['export']
    record_time = args['record_time']

    # DynAE, DynRNN, DynAERNN model params
    n_units, ae_units, rnn_units = [], [], []
    look_back, alpha = 0, 0
    if method in ['DynAE', 'DynRNN']:
        n_units = args['n_units']
    else:  # DynAERNN
        ae_units = args['ae_units']
        rnn_units = args['rnn_units']
    if method in ['DynAE', 'DynRNN', 'DynAERNN']:
        look_back = args['look_back']
        assert look_back > 0
    beta = args['beta']
    nu1 = args['nu1']
    nu2 = args['nu2']
    bias = args['bias']

    origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
    max_time_num = len(os.listdir(origin_base_path))
    node_path = os.path.abspath(os.path.join(base_path, node_file))
    nodes_set = pd.read_csv(node_path, names=['node'])
    node_num = nodes_set.shape[0]
    node_list = nodes_set['node'].tolist()
    data_loader = DataLoader(node_list, max_time_num, has_cuda=has_cuda)

    if start_idx < 0:
        start_idx = max_time_num + start_idx
    if end_idx < 0:  # original time range is [start_idx, end_idx] containing start_idx and end_idx
        end_idx = max_time_num + end_idx + 1
    else:
        end_idx = end_idx + 1

    assert start_idx + 1 - duration >= 0
    assert duration > look_back

    t1 = time.time()
    time_list = []

    print('start ' + method + ' embedding!')
    
    adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=0, duration=max_time_num, sep=file_sep, data_type='tensor')
    edge_list = [adj._indices() for adj in adj_list]  # edge_indices: [2, edge_num]
    neg_adj = data_loader.get_neg_adj(args['base_path'], args['neg_edges_file'], sep=file_sep, data_type='tensor')
    neg_edge_list = neg_adj._indices()  # neg_edge_indices: [2, neg_edge_num]
    nlabel_folder = args['nlabel_folder']
    nlabel_base_path = os.path.abspath(os.path.join(base_path, nlabel_folder))
    node_labels_train, node_labels_test, output_dim = data_loader.get_node_label_list(nlabel_base_path, start_idx=0, duration=max_time_num, sep=file_sep)

    embs_last, clss_last, lbl_last, embs_best, clss_best = [], [], [], [], []

    for idx in range(start_idx, end_idx):
        print('idx = ', idx)        
        
        embed_dim = args['embed_dim']
        cls_file = args.get('cls_file', None) + '-' + str(idx)
        cls_hidden_dim = args.get('cls_hid_dim', None)
        cls_layer_num = args.get('cls_layer_num', None)
        cls_bias = args.get('cls_bias', None)
        cls_activate_type = args.get('cls_activate_type', None)
        time_length = min(idx + duration, end_idx) - idx
        classifier = MLPClassifier(embed_dim, cls_hidden_dim, output_dim, layer_num=cls_layer_num, duration=time_length, bias=cls_bias, activate_type=cls_activate_type)
        
        # As DynAE, DynRNN, DynAERNN use original adjacent matrices as their input, so normalization is not necessary(normalization=Fals, add_eye=False) !
        adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx - duration + 1, duration=duration, sep=file_sep, normalize=False, add_eye=False, data_type='matrix')
        adj_list = [adj.tolil() for adj in adj_list]
        model = model_dict[method](input_dim=node_num, output_dim=embed_dim, look_back=look_back, n_units=n_units, ae_units=ae_units, rnn_units=rnn_units, bias=bias)

        loss = DynGraph2VecLoss(beta=beta, nu1=nu1, nu2=nu2)
        batch_generator = BatchGenerator(node_list=node_list, batch_size=batch_size, look_back=look_back, beta=beta, shuffle=shuffle, has_cuda=has_cuda)
        batch_predictor = BatchPredictor(node_list=node_list, batch_size=batch_size, has_cuda=has_cuda)
        trainer = DynamicEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=nodes_set['node'].tolist(), model=model, loss=loss,
                                             classifier=classifier, batch_generator=batch_generator, batch_predictor=batch_predictor, model_folder=model_folder, has_cuda=has_cuda)
        embedding_mat_last, cls_list_last, true_test_label, cost_time = trainer.learn_embedding(idx==start_idx,
                                            adj_list, node_labels_train[idx-time_length+1:idx+1], node_labels_test[idx-time_length+1:idx+1], edge_list, neg_edge_list, epoch=epoch, lr=lr, idx=idx, 
                                            model_file=model_file + '-' + str(idx), classifier_file=cls_file, seed=args.get('seed', None), load_model=load_model, export=export)
        embs_last.append(embedding_mat_last)
        clss_last.append(cls_list_last)
        lbl_last.append(true_test_label)
        time_list.append(cost_time)
    
    
    
    device = node_labels_train[0].device

    ################################################################################################
    print('start model evaluation!')
    cls_list = clss_last
    label_test = lbl_last
    output_list = embs_last

    num_class = cls_list[-1].size(1)
    all_results = {'Acc(micro)': [], 'Acc(macro)': [], 'Acc(weighted)': [],
                    'F1(micro)': [], 'F1(macro)': [], 'F1(weighted)': []}
    all_metrics = {}
    all_metrics['Acc(micro)'] = MulticlassAccuracy(num_classes=num_class, average='micro').to(device)
    all_metrics['Acc(macro)'] = MulticlassAccuracy(num_classes=num_class, average='macro').to(device)
    all_metrics['Acc(weighted)'] = MulticlassAccuracy(num_classes=num_class, average='weighted').to(device)
    all_metrics['F1(micro)'] = MulticlassF1Score(num_classes=num_class, average='micro').to(device)
    all_metrics['F1(macro)'] = MulticlassF1Score(num_classes=num_class, average='macro').to(device)
    all_metrics['F1(weighted)'] = MulticlassF1Score(num_classes=num_class, average='weighted').to(device)
    for pred, label in zip(cls_list, label_test):
        if label.size(0) == 0:
            continue
        for k, v in all_metrics.items():
            all_results[k].append(v(pred, label).item())
    for k, v in all_results.items():
        all_results[k] = np.mean(v)

    print('Test set results (last node):',
            '\n\tAcc(micro)={:.4f}'.format(all_results['Acc(micro)']), 'Acc(macro)={:.4f}'.format(all_results['Acc(macro)']),
            'Acc(weighted)={:.4f}'.format(all_results['Acc(weighted)']), '\n\tF1(micro)={:.4f}'.format(all_results['F1(micro)']),
            'F1(macro)={:.4f}'.format(all_results['F1(macro)']), 'F1(weighted)={:.4f}'.format(all_results['F1(weighted)']))
    
    # assert len(output_list) == len(edge_list)
    inductive_edge_scores, transductive_edge_scores = [], []
    for i in range(-3, 0):
        if (edge_list[i][0] == edge_list[i][1]).sum() > 0:
            mask = (edge_list[i][0] != edge_list[i][1])
            pos_edges = edge_list[i][:, mask]
        else:
            pos_edges = edge_list[i]
        neg_edges = neg_edge_list[:, torch.randperm(neg_edge_list.size(1))[:pos_edges.size(1)]]
        inductive_edge_scores.append(get_roc_scores(pos_edges, neg_edges, output_list[i].detach()))
        transductive_edge_scores.append(get_roc_scores(pos_edges, neg_edges, output_list[i-1].detach()))
    inductive_edge_scores = np.mean(inductive_edge_scores, axis=0)
    transductive_edge_scores = np.mean(transductive_edge_scores, axis=0)
    print('Test set results (inductive link):', 'F1={:.4f}'.format(inductive_edge_scores[0]), 'AUROC={:.4f}'.format(inductive_edge_scores[1]), 'AP={:.4f}'.format(inductive_edge_scores[2]))
    print('Test set results (transductive link):', 'F1={:.4f}'.format(transductive_edge_scores[0]), 'AUROC={:.4f}'.format(transductive_edge_scores[1]), 'AP={:.4f}'.format(transductive_edge_scores[2]))
    perf = {
        'Node-Acc(micro)': all_results['Acc(micro)'],
        'Node-Acc(macro)': all_results['Acc(macro)'],
        'Node-Acc(weighted)': all_results['Acc(weighted)'],
        'Node-F1(micro)': all_results['F1(micro)'],
        'Node-F1(macro)': all_results['F1(macro)'],
        'Node-F1(weighted)': all_results['F1(weighted)'],
        'Edge-F1(inductive)': inductive_edge_scores[0],
        'Edge-AUROC(inductive)': inductive_edge_scores[1],
        'Edge-AP(inductive)': inductive_edge_scores[2],
        'Edge-F1(transductive)': transductive_edge_scores[0],
        'Edge-AUROC(transductive)': transductive_edge_scores[1],
        'Edge-AP(transductive)': transductive_edge_scores[2],
    }
    with FileLock(os.path.join(trainer.embedding_base_path, 'perf_last.jsonl.lock')):
        with open(os.path.join(trainer.embedding_base_path,  'perf_last.jsonl'), 'a') as f:
            f.write(json.dumps(perf) + '\n')
    print('finish model evaluation!')
    ################################################################################################

    # record time cost of DynAE, DynRNN, DynAERNN
    if record_time:
        df_output = pd.DataFrame({'time': time_list})
        df_output.to_csv(os.path.join(base_path, method + '_time.csv'), sep=',', index=False)
    t2 = time.time()
    print('finish ' + method + ' embedding! cost time: ', t2 - t1, ' seconds!')
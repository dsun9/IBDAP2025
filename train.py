# coding: utf-8
import pandas as pd
import os
import time
import torch
import pickle
from embedding import SupervisedEmbedding, UnsupervisedEmbedding
from helper import DataLoader
from metrics import NegativeSamplingLoss, VAELoss, VAEClassificationLoss, NegativeSamplingClassificationLoss
from metrics import ClassificationLoss, VAEClassificationOwnLoss
from models import MLPClassifier, EdgeClassifier, InnerProduct
from utils import get_supported_gnn_methods


def get_data_loader(args):
    base_path = args['base_path']
    origin_folder = args['origin_folder']
    core_folder = args.get('core_folder', None)
    nfeature_folder = args.get('nfeature_folder', None)
    node_file = args['node_file']
    has_cuda = args['has_cuda']

    node_path = os.path.abspath(os.path.join(base_path, node_file))
    nodes_set = pd.read_csv(node_path, names=['node'])
    node_list = nodes_set['node'].tolist()
    node_num = nodes_set.shape[0]

    origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder)) if origin_folder else None
    core_base_path = os.path.abspath(os.path.join(base_path, core_folder)) if core_folder else None
    node_feature_path = os.path.abspath(os.path.join(base_path, nfeature_folder)) if nfeature_folder else None
    max_time_num = len(os.listdir(origin_base_path)) if origin_base_path else len(os.listdir(core_base_path))
    assert max_time_num > 0

    data_loader = DataLoader(node_list, max_time_num, has_cuda=has_cuda)
    args['origin_base_path'] = origin_base_path
    args['core_base_path'] = core_base_path
    args['nfeature_path'] = node_feature_path
    args['node_num'] = node_num
    return data_loader


def get_input_data(method, idx, time_length, data_loader, args):
    assert method in get_supported_gnn_methods()

    origin_base_path = args['origin_base_path']
    node_feature_path = args['nfeature_path']  # all the data sets we use don't have node features, so this path is None
    file_sep = args['file_sep']

    if method in ['GCRN']:  # If GCRN uses TgGCN, this GCRN should be removed!
        normalize, row_norm, add_eye = True, True, True
    elif method in ['EvolveGCN']:  # normalization is quite important for the performance improvement of EvolveGCN
        normalize, row_norm, add_eye = True, False, True
    else:
        normalize, row_norm, add_eye = False, False, False

    adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=time_length, sep=file_sep, normalize=normalize, row_norm=row_norm, add_eye=add_eye, data_type='tensor')
    # all gnn methods need edge_list when learning_type='S-link'
    edge_list = [adj._indices() for adj in adj_list]  # edge_indices: [2, edge_num]
    
    neg_adj = data_loader.get_neg_adj(args['base_path'], args['neg_edges_file'], sep=file_sep, data_type='tensor')
    neg_edge_list = neg_adj._indices()  # neg_edge_indices: [2, neg_edge_num]

    if method in ['EvolveGCN'] and node_feature_path is None:
        init_type = args['init_type']
        std = args.get('std', 1e-4)
        x_list, input_dim = data_loader.get_degree_feature_list(origin_base_path, start_idx=idx, duration=time_length, sep=file_sep, init_type=init_type, std=std)
    else:
        x_list, input_dim = data_loader.get_feature_list(node_feature_path, start_idx=idx, duration=time_length, shuffle=False)
        if method == 'VGRNN' or method == 'RIVGAE':
            x_list = torch.stack(x_list)

    node_dist_list = None
    return input_dim, adj_list, x_list, edge_list, node_dist_list, neg_edge_list


def get_gnn_model(method, time_length, args):
    assert method in get_supported_gnn_methods()

    from method.gcrn import GCRN
    from method.egcn import EvolveGCN
    from method.vgrnn import VGRNN
    from method.rivgae import RIVGAE

    input_dim = args['input_dim']
    hidden_dim = args['hid_dim']
    embed_dim = args['embed_dim']
    dropout = args.get('dropout', None)
    bias = args.get('bias', None)

    if method in ['GCRN']:
        feature_pre = args['feature_pre']
        feature_dim = args['feature_dim']
        layer_num = args['layer_num']
        rnn_type = args['rnn_type']
        return GCRN(input_dim, feature_dim, hidden_dim, embed_dim, feature_pre=feature_pre, layer_num=layer_num, dropout=dropout, bias=bias,
                    duration=time_length, rnn_type=rnn_type)
    elif method == 'VGRNN':
        rnn_layer_num = args['rnn_layer_num']
        conv_type = args['conv_type']
        return VGRNN(input_dim, hidden_dim, embed_dim, rnn_layer_num=rnn_layer_num, conv_type=conv_type, bias=bias)
    elif method == 'RIVGAE':
        rnn_layer_num = args['rnn_layer_num']
        conv_type = args['conv_type']
        with open(os.path.join(args['base_path'], args['meta_file']), 'rb') as f:
            meta = pickle.load(f)
        num_user = len(meta['id2act'])
        num_assertion = len(meta['id2asser'])
        print('num_user: ', num_user, ', num_assertion: ', num_assertion)
        return RIVGAE(input_dim, hidden_dim, embed_dim, rnn_layer_num=rnn_layer_num, num_user=num_user, num_assertion=num_assertion, conv_type=conv_type, bias=bias)
    elif method == 'EvolveGCN':
        egcn_type = args['model_type']
        return EvolveGCN(input_dim, hidden_dim, embed_dim, egcn_type=egcn_type)
    else:
        assert False


def get_loss(method, idx, time_length, data_loader, args):
    learning_type = args['learning_type']
    assert learning_type in ['U-neg', 'U-own', 'U-neg-S-node', 'U-neg-S-node-only', 'S-node', 'S-edge']
    base_path = args['base_path']
    file_sep = args['file_sep']

    if learning_type == 'U-neg':
        walk_pair_folder = args['walk_pair_folder']
        node_freq_folder = args['node_freq_folder']
        neg_num = args['neg_num']
        Q = args['Q']
        walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
        node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))
        node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=time_length)
        neg_freq_list = data_loader.get_node_freq_list(node_freq_base_path, start_idx=idx, duration=time_length)
        loss = NegativeSamplingLoss(node_pair_list, neg_freq_list, neg_num=neg_num, Q=Q)
        return loss
    elif learning_type == 'U-own':
        if method == 'VGRNN' or method == 'RIVGAE':
            eps = args['eps']
            loss = VAELoss(eps=eps)
        else:
            raise NotImplementedError('No implementation of ' + method + '\'s unsupervised learning loss!')
        return loss
    else:   # supervised learning_type ['S-node', 'S-edge']:
        embed_dim = args['embed_dim']
        cls_hidden_dim = args.get('cls_hid_dim', None)
        cls_layer_num = args.get('cls_layer_num', None)
        cls_bias = args.get('cls_bias', None)
        cls_activate_type = args.get('cls_activate_type', None)

        node_label_list, edge_label_list = None, None
        if learning_type == 'S-node':
            nlabel_folder = args['nlabel_folder']
            nlabel_base_path = os.path.abspath(os.path.join(base_path, nlabel_folder))
            node_label_list_train, node_label_list_test, output_dim = data_loader.get_node_label_list(nlabel_base_path, start_idx=idx, duration=time_length, sep=file_sep)
            classifier = MLPClassifier(embed_dim, cls_hidden_dim, output_dim, layer_num=cls_layer_num, duration=time_length, bias=cls_bias, activate_type=cls_activate_type)
        elif learning_type == 'S-edge':
            elabel_folder = args['elabel_folder']
            elabel_base_path = os.path.abspath(os.path.join(base_path, elabel_folder))
            edge_label_list, output_dim = data_loader.get_edge_label_list(elabel_base_path, start_idx=idx, duration=time_length, sep=file_sep)
            classifier = EdgeClassifier(embed_dim, cls_hidden_dim, output_dim, layer_num=cls_layer_num, duration=time_length, bias=cls_bias, activate_type=cls_activate_type)
        elif learning_type == 'U-neg-S-node' or learning_type == 'U-neg-S-node-only':
            nlabel_folder = args['nlabel_folder']
            nlabel_base_path = os.path.abspath(os.path.join(base_path, nlabel_folder))
            node_label_list_train, node_label_list_test, output_dim = data_loader.get_node_label_list(nlabel_base_path, start_idx=idx, duration=time_length, sep=file_sep)
            classifier = MLPClassifier(embed_dim, cls_hidden_dim, output_dim, layer_num=cls_layer_num, duration=time_length, bias=cls_bias, activate_type=cls_activate_type)
            walk_pair_folder = args['walk_pair_folder']
            node_freq_folder = args['node_freq_folder']
            neg_num = args['neg_num']
            Q = args['Q']
            walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
            node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))
            node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=time_length)
            neg_freq_list = data_loader.get_node_freq_list(node_freq_base_path, start_idx=idx, duration=time_length)
        else:  # S-link-st, S-link-dy
            classifier = InnerProduct()
            output_dim = 2  # postive link & negative link
        # loss
        if method == 'VGRNN':
            eps = args['eps']
            loss = VAEClassificationLoss(output_dim, eps=eps)
        elif method == 'RIVGAE':
            eps = args['eps']
            with open(os.path.join(args['base_path'], args['meta_file']), 'rb') as f:
                meta = pickle.load(f)
            num_user = len(meta['id2act'])
            num_assertion = len(meta['id2asser'])
            loss = VAEClassificationOwnLoss(output_dim, num_user, num_assertion, eps=eps)
        elif method in ['EvolveGCN', 'GCRN'] and (learning_type == 'U-neg-S-node' or learning_type == 'U-neg-S-node-only'):
            loss = NegativeSamplingClassificationLoss(learning_type == 'U-neg-S-node', output_dim, node_pair_list, neg_freq_list, neg_num=neg_num, Q=Q)
        else:
            loss = ClassificationLoss(output_dim)
        return loss, classifier, node_label_list_train, node_label_list_test, edge_label_list


def gnn_embedding(method, args):
    # common params
    base_path = args['base_path']
    origin_folder = args['origin_folder']
    embedding_folder = args['embed_folder']
    model_folder = args['model_folder']
    model_file = args['model_file']
    node_file = args['node_file']
    # file_sep = args['file_sep']
    start_idx = args['start_idx']
    end_idx = args['end_idx']
    duration = args['duration']
    has_cuda = args['has_cuda']
    learning_type = args['learning_type']
    # hidden_dim = args['hid_dim']
    # embed_dim = args['embed_dim']
    epoch = args['epoch']
    lr = args['lr']
    batch_size = args['batch_size']
    load_model = args['load_model']
    shuffle = args['shuffle']
    export = args['export']
    record_time = args['record_time']
    weight_decay = args['weight_decay']

    data_loader = get_data_loader(args)
    max_time_num = data_loader.max_time_num
    node_list = data_loader.full_node_list

    if start_idx < 0:
        start_idx = max_time_num + start_idx
    if end_idx < 0:  # original time range is [start_idx, end_idx] containing start_idx and end_idx
        end_idx = max_time_num + end_idx + 1
    else:
        end_idx = end_idx + 1
    step = duration
    if learning_type == 'S-link-dy':
        assert duration >= 2 and end_idx - start_idx >= 1
        end_idx = end_idx - 1
        step = duration - 1  # -1 is to make step and end_idx adapt to the dynamic link prediction setting

    t1 = time.time()
    time_list = []
    print('start_idx = ', start_idx, ', end_idx = ', end_idx, ', duration = ', duration)
    print('start ' + method + ' embedding!')
    for idx in range(start_idx, end_idx, step):
        print('idx = ', idx, ', duration = ', duration)
        time_length = min(idx + duration, end_idx) - idx
        input_dim, adj_list, x_list, edge_list, node_dist_list, neg_edge_list = get_input_data(method, idx, time_length, data_loader, args)
        args['input_dim'] = input_dim
        model = get_gnn_model(method, time_length, args)

        if learning_type in ['U-neg', 'U-own']:
            loss = get_loss(method, idx, time_length, data_loader, args)
            trainer = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=node_list,
                                            model=model, loss=loss, model_folder=model_folder, has_cuda=has_cuda)
            cost_time = trainer.learn_embedding(adj_list, x_list, edge_list, node_dist_list, epoch=epoch, batch_size=batch_size, lr=lr, start_idx=idx, weight_decay=weight_decay,
                                                model_file=model_file, load_model=load_model, shuffle=shuffle, export=export)
            time_list.append(cost_time)

        else:  # supervised learning
            cls_file = args.get('cls_file', None)
            train_ratio = args['train_ratio']
            val_ratio = args['val_ratio']
            test_ratio = args['test_ratio']
            loss, classifier, node_labels_train, node_labels_test, edge_labels = get_loss(method, idx, time_length, data_loader, args)
            trainer = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=node_list, model=model,
                                          loss=loss, classifier=classifier, model_folder=model_folder, has_cuda=has_cuda)
            cost_time = trainer.learn_embedding(adj_list, x_list, node_labels_train, node_labels_test, edge_labels, edge_list, neg_edge_list, node_dist_list, learning_type=learning_type, epoch=epoch, batch_size=batch_size,
                                                lr=lr, start_idx=idx, weight_decay=weight_decay, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                                                model_file=model_file, classifier_file=cls_file, seed=args.get('seed', None), load_model=load_model, shuffle=shuffle, export=export, B=args.get('B', None), args=args)
            time_list.append(cost_time)

    # record time cost of the model
    if record_time:
        df_output = pd.DataFrame({'time': time_list})
        df_output.to_csv(os.path.join(base_path, method + '_time.csv'), sep=',', index=False)
    t2 = time.time()
    print('finish ' + method + ' embedding! cost time: ', t2 - t1, ' seconds!')

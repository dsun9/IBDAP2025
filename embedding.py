# coding: utf-8
import json
import numpy as np
import pandas as pd
import os
import gc
import time
import torch
from filelock import FileLock
from models import MLPClassifier
from utils import check_and_make_path, get_roc_scores

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)


# The base class of embedding
class BaseEmbedding:
    base_path: str
    origin_base_path: str
    embedding_base_path: str
    model_base_path: str
    file_sep: str
    full_node_list: list
    node_num: int
    timestamp_list: list
    has_cuda: bool
    device: torch.device

    def __init__(self, base_path, origin_folder, embedding_folder, node_list, model, loss, model_folder='model', file_sep='\t', has_cuda=False):
        # file paths
        self.base_path = base_path
        self.origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
        self.embedding_base_path = os.path.abspath(os.path.join(base_path, embedding_folder))
        self.model_base_path = os.path.abspath(os.path.join(base_path, model_folder))
        self.has_cuda = has_cuda
        self.device = torch.device('cuda:0') if has_cuda else torch.device('cpu')
        self.model = model
        self.loss = loss

        self.file_sep = file_sep
        self.full_node_list = node_list
        self.node_num = len(self.full_node_list)  # node num
        self.timestamp_list = sorted(os.listdir(self.origin_base_path))

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.model_base_path)

    def clear_cache(self):
        if self.has_cuda:
            torch.cuda.empty_cache()
        else:
            gc.collect()

    def prepare(self, load_model, model_file, classifier_file=None, lr=1e-3, weight_decay=0.):
        classifier = self.classifier if hasattr(self, 'classifier') else None

        if load_model:
            model_path = os.path.join(self.model_base_path, model_file)
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(os.path.join(self.model_base_path, model_file)))
                self.model.eval()
            if classifier_file and classifier:
                classifier_path = os.path.join(self.model_base_path, classifier_file)
                classifier.load_state_dict(torch.load(classifier_path))
                classifier.eval()

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        if classifier:
            classifier = classifier.to(self.device)

        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.zero_grad()
        return self.model, self.loss, optimizer, classifier

    def get_batch_info(self, **kwargs):
        pass

    def get_model_res(self, **kwargs):
        pass

    def save_embedding(self, output_list, start_idx):
        if isinstance(output_list, torch.Tensor) and len(output_list.size()) == 2:  # static embedding
            embedding = output_list
            output_list = [embedding]
        # output_list supports two type: list and torch.Tensor(2d or 3d tensor)
        for i in range(len(output_list)):
            embedding = output_list[i]
            timestamp = self.timestamp_list[start_idx + i].split('.')[0]
            df_export = pd.DataFrame(data=embedding.cpu().detach().numpy(), index=self.full_node_list)
            embedding_path = os.path.join(self.embedding_base_path, timestamp + '.csv')
            df_export.to_csv(embedding_path, sep=self.file_sep, header=True, index=True)


# Supervised embedding class(used for node classification)
class SupervisedEmbedding(BaseEmbedding):

    def __init__(self, base_path, origin_folder, embedding_folder, node_list, model, loss, classifier: MLPClassifier, model_folder='model', has_cuda=False):
        super(SupervisedEmbedding, self).__init__(base_path, origin_folder, embedding_folder, node_list, model, loss, model_folder=model_folder, has_cuda=has_cuda)
        self.classifier = classifier

    def get_batch_info(self, learning_type, node_labels_train, node_labels_test, edge_labels, edge_list, batch_size, shuffle, train_ratio, val_ratio, test_ratio):
        # consider node classification data / edge classification data
        if learning_type in ['S-node', 'S-edge', 'U-neg-S-node', 'U-neg-S-node-only']:
            batch_num = self.node_num // batch_size
            if self.node_num % batch_size != 0:
                batch_num += 1
            if learning_type == 'S-node' or learning_type == 'U-neg-S-node' or learning_type == 'U-neg-S-node-only':
                assert node_labels_train
                assert node_labels_test
                timestamp_num = len(node_labels_train)
                assert len(node_labels_train) == len(node_labels_test)
                device = node_labels_train[0].device
            else:
                assert edge_labels
                timestamp_num = len(edge_labels)
                device = edge_labels[0].device

            idx_train, label_train, idx_val, label_val, idx_test, label_test, train_cross_mat = [], [], [], [], [], [], []
            for i in range(timestamp_num):
                if learning_type == 'S-node' or learning_type == 'U-neg-S-node' or learning_type == 'U-neg-S-node-only':
                    cur_labels_train = node_labels_train[i]  # tensor
                    cur_labels_test = node_labels_test[i]  # tensor
                    assert cur_labels_train.shape[1] == 2
                    assert cur_labels_test.shape[1] == 2
                else:
                    cur_labels = edge_labels[i]  # tensor
                    assert cur_labels.shape[1] == 3

                if learning_type == 'S-node' or learning_type == 'U-neg-S-node' or learning_type == 'U-neg-S-node-only':
                    train_items, train_labels = cur_labels_train[:, 0], cur_labels_train[:, 1]
                    val_items, val_labels = cur_labels_test[:, 0], cur_labels_test[:, 1]
                    test_items, test_labels = cur_labels_test[:, 0], cur_labels_test[:, 1]
                else:
                    assert False
                cross_mat = torch.zeros((train_items.size(0), train_items.size(0)), device=device)
                for i, v1 in enumerate(train_items.cpu().tolist()):
                    for j, v2 in enumerate(train_items.cpu().tolist()):
                        if train_labels[i] == train_labels[j]:
                            cross_mat[i][j] = 1
                idx_train.append(train_items)
                label_train.append(train_labels)
                idx_val.append(val_items)
                label_val.append(val_labels)
                idx_test.append(test_items)
                label_test.append(test_labels)
                train_cross_mat.append(cross_mat)
            return idx_train, label_train, idx_val, label_val, idx_test, label_test, train_cross_mat, batch_num
        else:
            assert False

    def get_model_res(self, learning_type, adj_list, x_list, edge_list, node_dist_list, batch_indices, model, classifier, hx=None, train_cross_mat=None, B=None):
        if model.method_name == 'VGRNN':
            embedding_list, _, loss_data_list = model(x_list, edge_list, hx)
            embedding_list = embedding_list[:-1] if learning_type == 'S-link-dy' else embedding_list
            cls_list = classifier(embedding_list, batch_indices)
            loss_input_list = loss_data_list
            loss_input_list.append(adj_list)
            loss_input_list.append(cls_list)
            embedding_list = embedding_list[:-1] if learning_type == 'S-link-dy' else embedding_list
        elif model.method_name == 'RIVGAE':
            embedding_list, _, loss_data_list = model(x_list, edge_list, hx, B)
            embedding_list = embedding_list[:-1] if learning_type == 'S-link-dy' else embedding_list
            cls_list = classifier(embedding_list, batch_indices)
            loss_input_list = loss_data_list
            loss_input_list.append(adj_list)
            loss_input_list.append([embedding_list[i][x] for i, x in enumerate(batch_indices)])
            loss_input_list.append(train_cross_mat)
            loss_input_list.append(cls_list)
            embedding_list = embedding_list[:-1] if learning_type == 'S-link-dy' else embedding_list
        else:  # GCRN, EvolveGCN
            embedding_list = model(x_list, adj_list)
            embedding_list = embedding_list[:-1] if learning_type == 'S-link-dy' else embedding_list
            cls_list = classifier(embedding_list, batch_indices)
            loss_input_list = [embedding_list, cls_list]
        output_list = embedding_list
        return loss_input_list, output_list, hx

    # edge_list parameter is only used by VGRNN, node_dist_list parameter is only used by PGNN
    # node_labels parameter is used for node classification, edge_labels parameter is used for edge classification
    def learn_embedding(self, adj_list, x_list, node_labels_train=None, node_labels_test=None, edge_labels=None, edge_list=None, neg_edge_list=None, node_dist_list=None, learning_type='S-node', epoch=50, batch_size=1024, lr=1e-3, start_idx=0, weight_decay=0.,
                        train_ratio=0.5, val_ratio=0.3, test_ratio=0.2, model_file='rivgae', classifier_file='rivgae_cls', seed=None, load_model=False, shuffle=True, export=True, B=None, args=None):
        assert train_ratio + val_ratio + test_ratio <= 1.0
        if seed is not None:
            model_file = model_file + '_' + str(seed)
            classifier_file = classifier_file + '_' + str(seed)
        if B is not None:
            B = torch.FloatTensor(B).to(self.device)
        else:
            B = torch.eye(args['embed_dim'], device=self.device)
        # prepare model, loss model, optimizer and classifier model
        model, loss_model, optimizer, classifier = self.prepare(load_model, model_file, classifier_file, lr, weight_decay)
        idx_train, label_train, idx_val, label_val, idx_test, label_test, train_cross_mat, batch_num = self.get_batch_info(learning_type, node_labels_train, node_labels_test, edge_labels, edge_list, batch_size, shuffle, train_ratio, val_ratio, test_ratio)
        all_nodes = torch.arange(self.node_num, device=self.device)
        self.clear_cache()
        # time.sleep(100)
        best_acc, best_auc, best_hx = 0, 0, None
        print('start supervised training!')
        st = time.time()
        model.train()
        for i in range(epoch):
            node_indices = all_nodes[torch.randperm(self.node_num)] if shuffle else all_nodes  # Tensor
            # batch_indices = node_indices[:min(self.node_num, batch_size)]
            hx = None  # used for VGRNN
            
            if learning_type == 'U-neg-S-node' or learning_type == 'U-neg-S-node-only':
                loss_sum = 0
                t1 = time.time()
                for j in range(batch_num):
                    batch_indices = node_indices[j * batch_size: min(self.node_num, (j + 1) * batch_size)]
                    loss_input_list, output_list, hx = self.get_model_res(learning_type, adj_list, x_list, edge_list, node_dist_list, idx_train, model, classifier, hx, train_cross_mat, B=B)
                    loss_input_list.append(batch_indices)
                    loss_train, acc_train, auc_train = loss_model(loss_input_list, label_train)
                    loss_train.backward()
                    loss_sum += loss_train.item()
                    # gradient accumulation
                    if j == batch_num - 1:
                        optimizer.step()  # update gradient
                        model.zero_grad()
                    self.clear_cache()
                    if j < batch_num - 1:
                        print('Epoch: ' + str(i + 1), ', batch num: ' + str(j + 1), 'loss_train: {:.4f}'.format(loss_train.item()))
                if i == 0:
                    print('Epoch: ' + str(i + 1), 'loss_train: {:.4f}'.format(loss_sum))
                else:
                    loss_input_list, output_list, hx = self.get_model_res(learning_type, adj_list, x_list, edge_list, node_dist_list, idx_val, model, classifier, hx, B=B)
                    loss_input_list.append(None)
                    loss_val, acc_val, auc_val = loss_model(loss_input_list, label_val)
                    print('Epoch: ' + str(i + 1), 'loss_train: {:.4f}'.format(loss_sum), 'acc_train: {:.4f}'.format(acc_train.item()), 'auc_train: {:.4f}'.format(auc_train),
                        'loss_val: {:.4f}'.format(loss_val.item()), 'acc_val: {:.4f}'.format(acc_val.item()), 'auc_val: {:.4f}'.format(auc_val),  'cost time: {:.4f}s'.format(time.time() - t1))
                    if acc_val > best_acc or (acc_val == best_acc and auc_val > best_auc):
                        best_acc = acc_val
                        best_auc = auc_val
                        best_hx = hx
                        if model_file:
                            torch.save(model.state_dict(), os.path.join(self.model_base_path, model_file))
                        if classifier_file:
                            torch.save(classifier.state_dict(), os.path.join(self.model_base_path, classifier_file))
                self.clear_cache()
            
            else:
                t1 = time.time()
                loss_input_list, output_list, hx = self.get_model_res(learning_type, adj_list, x_list, edge_list, node_dist_list, idx_train, model, classifier, hx, train_cross_mat, B=B)
                loss_train, acc_train, auc_train = loss_model(loss_input_list, label_train)
                loss_train.backward()
                optimizer.step()  # update gradient
                model.zero_grad()
                # validation
                if i == 0:
                    print('Epoch: ' + str(i + 1), 'loss_train: {:.4f}'.format(loss_train.item()))
                else:
                    loss_input_list, output_list, hx = self.get_model_res(learning_type, adj_list, x_list, edge_list, node_dist_list, idx_val, model, classifier, hx, B=B)
                    loss_val, acc_val, auc_val = loss_model(loss_input_list, label_val)
                    print('Epoch: ' + str(i + 1), 'loss_train: {:.4f}'.format(loss_train.item()), 'acc_train: {:.4f}'.format(acc_train.item()), 'auc_train: {:.4f}'.format(auc_train),
                        'loss_val: {:.4f}'.format(loss_val.item()), 'acc_val: {:.4f}'.format(acc_val.item()), 'auc_val: {:.4f}'.format(auc_val),  'cost time: {:.4f}s'.format(time.time() - t1))
                    if acc_val > best_acc or (acc_val == best_acc and auc_val > best_auc):
                        best_acc = acc_val
                        best_auc = auc_val
                        best_hx = hx
                        if model_file:
                            torch.save(model.state_dict(), os.path.join(self.model_base_path, model_file))
                        if classifier_file:
                            torch.save(classifier.state_dict(), os.path.join(self.model_base_path, classifier_file))
                self.clear_cache()
        print('finish supervised training!')

        model.eval()
        classifier.eval()
        loss_model.disable()
        ################################################################################################
        print('start model evaluation!')
        loss_input_list, output_list, _ = self.get_model_res(learning_type, adj_list, x_list, edge_list, node_dist_list, idx_test, model, classifier, best_hx, B=B)
        
        cls_list = loss_input_list[-1]
        if learning_type == 'U-neg-S-node' or learning_type == 'U-neg-S-node-only':
            loss_input_list.append(None)
        num_class = cls_list[-1].size(1)
        all_results = {'Acc(micro)': [], 'Acc(macro)': [], 'Acc(weighted)': [],
                       'F1(micro)': [], 'F1(macro)': [], 'F1(weighted)': []}
        all_metrics = {}
        all_metrics['Acc(micro)'] = MulticlassAccuracy(num_classes=num_class, average='micro').to(self.device)
        all_metrics['Acc(macro)'] = MulticlassAccuracy(num_classes=num_class, average='macro').to(self.device)
        all_metrics['Acc(weighted)'] = MulticlassAccuracy(num_classes=num_class, average='weighted').to(self.device)
        all_metrics['F1(micro)'] = MulticlassF1Score(num_classes=num_class, average='micro').to(self.device)
        all_metrics['F1(macro)'] = MulticlassF1Score(num_classes=num_class, average='macro').to(self.device)
        all_metrics['F1(weighted)'] = MulticlassF1Score(num_classes=num_class, average='weighted').to(self.device)
        for pred, label in zip(cls_list, label_test):
            if label.size(0) == 0:
                continue
            for k, v in all_metrics.items():
                all_results[k].append(v(pred, label).item())
        for k, v in all_results.items():
            all_results[k] = np.mean(v)

        loss_test, acc_test, auc_test = loss_model(loss_input_list, label_test)
        print('Test set results (last node):', 'loss={:.4f}'.format(loss_test.item()), '\n\taccuracy={:.4f}'.format(acc_test.item()), 'auc={:.4f}'.format(auc_test.item()),
              '\n\tAcc(micro)={:.4f}'.format(all_results['Acc(micro)']), 'Acc(macro)={:.4f}'.format(all_results['Acc(macro)']),
              'Acc(weighted)={:.4f}'.format(all_results['Acc(weighted)']), '\n\tF1(micro)={:.4f}'.format(all_results['F1(micro)']),
              'F1(macro)={:.4f}'.format(all_results['F1(macro)']), 'F1(weighted)={:.4f}'.format(all_results['F1(weighted)']))
        
        assert len(output_list) == len(edge_list)
        inductive_edge_scores, transductive_edge_scores = [], []
        inductive_edge_scores_b, transductive_edge_scores_b = [], []
        for i in range(len(edge_list) - 3, len(edge_list)):
            if (edge_list[i][0] == edge_list[i][1]).sum() > 0:
                mask = (edge_list[i][0] != edge_list[i][1])
                pos_edges = edge_list[i][:, mask]
            else:
                pos_edges = edge_list[i]
            neg_edges = neg_edge_list[:, torch.randperm(neg_edge_list.size(1))[:pos_edges.size(1)]]
            inductive_edge_scores.append(get_roc_scores(pos_edges, neg_edges, output_list[i].detach()))
            transductive_edge_scores.append(get_roc_scores(pos_edges, neg_edges, output_list[i-1].detach()))
            inductive_edge_scores_b.append(get_roc_scores(pos_edges, neg_edges, output_list[i].detach(), B))
            transductive_edge_scores_b.append(get_roc_scores(pos_edges, neg_edges, output_list[i-1].detach(), B))
        inductive_edge_scores = np.mean(inductive_edge_scores, axis=0)
        transductive_edge_scores = np.mean(transductive_edge_scores, axis=0)
        inductive_edge_scores_b = np.mean(inductive_edge_scores_b, axis=0)
        transductive_edge_scores_b = np.mean(transductive_edge_scores_b, axis=0)
        print('Test set results (inductive link):', 'F1={:.4f}'.format(inductive_edge_scores[0]), 'AUROC={:.4f}'.format(inductive_edge_scores[1]), 'AP={:.4f}'.format(inductive_edge_scores[2]))
        print('Test set results (transductive link):', 'F1={:.4f}'.format(transductive_edge_scores[0]), 'AUROC={:.4f}'.format(transductive_edge_scores[1]), 'AP={:.4f}'.format(transductive_edge_scores[2]))
        print('Test set results (inductive link) B:', 'F1={:.4f}'.format(inductive_edge_scores_b[0]), 'AUROC={:.4f}'.format(inductive_edge_scores_b[1]), 'AP={:.4f}'.format(inductive_edge_scores_b[2]))
        print('Test set results (transductive link) B:', 'F1={:.4f}'.format(transductive_edge_scores_b[0]), 'AUROC={:.4f}'.format(transductive_edge_scores_b[1]), 'AP={:.4f}'.format(transductive_edge_scores_b[2]))
        
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
            'Edge-F1(inductive)-B': inductive_edge_scores_b[0],
            'Edge-AUROC(inductive)-B': inductive_edge_scores_b[1],
            'Edge-AP(inductive)-B': inductive_edge_scores_b[2],
            'Edge-F1(transductive)-B': transductive_edge_scores_b[0],
            'Edge-AUROC(transductive)-B': transductive_edge_scores_b[1],
            'Edge-AP(transductive)-B': transductive_edge_scores_b[2],
        }
        with FileLock(os.path.join(self.embedding_base_path, 'perf_last.jsonl.lock')):
            with open(os.path.join(self.embedding_base_path,  'perf_last.jsonl'), 'a') as f:
                f.write(json.dumps(perf) + '\n')
        print('finish model evaluation!')
        self.clear_cache()
        ################################################################################################

        # load embedding model and classifier model
        if model_file:
            model.load_state_dict(torch.load(os.path.join(self.model_base_path, model_file)))
            model.eval()
        if classifier_file:
            classifier.load_state_dict(torch.load(os.path.join(self.model_base_path, classifier_file)))
            classifier.eval()
            
        ################################################################################################
        print('start model evaluation!')
        loss_input_list, output_list, _ = self.get_model_res(learning_type, adj_list, x_list, edge_list, node_dist_list, idx_test, model, classifier, best_hx, B=B)
        
        cls_list = loss_input_list[-1]
        if learning_type == 'U-neg-S-node' or learning_type == 'U-neg-S-node-only':
            loss_input_list.append(None)
        num_class = cls_list[-1].size(1)
        all_metrics = {}
        all_results = {'Acc(micro)': [], 'Acc(macro)': [], 'Acc(weighted)': [],
                       'F1(micro)': [], 'F1(macro)': [], 'F1(weighted)': []}
        all_metrics['Acc(micro)'] = MulticlassAccuracy(num_classes=num_class, average='micro').to(self.device)
        all_metrics['Acc(macro)'] = MulticlassAccuracy(num_classes=num_class, average='macro').to(self.device)
        all_metrics['Acc(weighted)'] = MulticlassAccuracy(num_classes=num_class, average='weighted').to(self.device)
        all_metrics['F1(micro)'] = MulticlassF1Score(num_classes=num_class, average='micro').to(self.device)
        all_metrics['F1(macro)'] = MulticlassF1Score(num_classes=num_class, average='macro').to(self.device)
        all_metrics['F1(weighted)'] = MulticlassF1Score(num_classes=num_class, average='weighted').to(self.device)
        for pred, label in zip(cls_list, label_test):
            if label.size(0) == 0:
                continue
            for k, v in all_metrics.items():
                all_results[k].append(v(pred, label).item())
        for k, v in all_results.items():
            all_results[k] = np.mean(v)

        loss_test, acc_test, auc_test = loss_model(loss_input_list, label_test)
        print('Test set results (best node):', 'loss={:.4f}'.format(loss_test.item()), '\n\taccuracy={:.4f}'.format(acc_test.item()), 'auc={:.4f}'.format(auc_test.item()),
              '\n\tAcc(micro)={:.4f}'.format(all_results['Acc(micro)']), 'Acc(macro)={:.4f}'.format(all_results['Acc(macro)']),
              'Acc(weighted)={:.4f}'.format(all_results['Acc(weighted)']), '\n\tF1(micro)={:.4f}'.format(all_results['F1(micro)']),
              'F1(macro)={:.4f}'.format(all_results['F1(macro)']), 'F1(weighted)={:.4f}'.format(all_results['F1(weighted)']))
        
        assert len(output_list) == len(edge_list)
        inductive_edge_scores, transductive_edge_scores = [], []
        inductive_edge_scores_b, transductive_edge_scores_b = [], []
        for i in range(len(edge_list) - 3, len(edge_list)):
            if (edge_list[i][0] == edge_list[i][1]).sum() > 0:
                mask = (edge_list[i][0] != edge_list[i][1])
                pos_edges = edge_list[i][:, mask]
            else:
                pos_edges = edge_list[i]
            neg_edges = neg_edge_list[:, torch.randperm(neg_edge_list.size(1))[:pos_edges.size(1)]]
            inductive_edge_scores.append(get_roc_scores(pos_edges, neg_edges, output_list[i].detach()))
            transductive_edge_scores.append(get_roc_scores(pos_edges, neg_edges, output_list[i-1].detach()))
            inductive_edge_scores_b.append(get_roc_scores(pos_edges, neg_edges, output_list[i].detach(), B))
            transductive_edge_scores_b.append(get_roc_scores(pos_edges, neg_edges, output_list[i-1].detach(), B))
        inductive_edge_scores = np.mean(inductive_edge_scores, axis=0)
        transductive_edge_scores = np.mean(transductive_edge_scores, axis=0)
        inductive_edge_scores_b = np.mean(inductive_edge_scores_b, axis=0)
        transductive_edge_scores_b = np.mean(transductive_edge_scores_b, axis=0)
        print('Test set results (inductive link):', 'F1={:.4f}'.format(inductive_edge_scores[0]), 'AUROC={:.4f}'.format(inductive_edge_scores[1]), 'AP={:.4f}'.format(inductive_edge_scores[2]))
        print('Test set results (transductive link):', 'F1={:.4f}'.format(transductive_edge_scores[0]), 'AUROC={:.4f}'.format(transductive_edge_scores[1]), 'AP={:.4f}'.format(transductive_edge_scores[2]))
        print('Test set results (inductive link) B:', 'F1={:.4f}'.format(inductive_edge_scores_b[0]), 'AUROC={:.4f}'.format(inductive_edge_scores_b[1]), 'AP={:.4f}'.format(inductive_edge_scores_b[2]))
        print('Test set results (transductive link) B:', 'F1={:.4f}'.format(transductive_edge_scores_b[0]), 'AUROC={:.4f}'.format(transductive_edge_scores_b[1]), 'AP={:.4f}'.format(transductive_edge_scores_b[2]))
        
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
            'Edge-F1(inductive)-B': inductive_edge_scores_b[0],
            'Edge-AUROC(inductive)-B': inductive_edge_scores_b[1],
            'Edge-AP(inductive)-B': inductive_edge_scores_b[2],
            'Edge-F1(transductive)-B': transductive_edge_scores_b[0],
            'Edge-AUROC(transductive)-B': transductive_edge_scores_b[1],
            'Edge-AP(transductive)-B': transductive_edge_scores_b[2],
        }
        with FileLock(os.path.join(self.embedding_base_path, 'perf_best.jsonl.lock')):
            with open(os.path.join(self.embedding_base_path,  'perf_best.jsonl'), 'a') as f:
                f.write(json.dumps(perf) + '\n')
        print('finish model evaluation!')
        ################################################################################################
            
        en = time.time()
        cost_time = en - st

        if export:
            self.save_embedding(output_list, start_idx)
        del adj_list, x_list, output_list, model
        self.clear_cache()
        print('training total time: ', cost_time, ' seconds!')
        return cost_time


# Unsupervised embedding class
class UnsupervisedEmbedding(BaseEmbedding):
    def __init__(self, base_path, origin_folder, embedding_folder, node_list, model, loss, model_folder='model', has_cuda=False):
        super(UnsupervisedEmbedding, self).__init__(base_path, origin_folder, embedding_folder, node_list, model, loss, model_folder=model_folder, has_cuda=has_cuda)

    def get_model_res(self, adj_list, x_list, edge_list, node_dist_list, model, batch_indices, hx):
        if model.method_name == 'VGRNN':
            embedding_list, hx, loss_data_list = model(x_list, edge_list, hx)
            loss_input_list = loss_data_list
            loss_input_list.append(adj_list)
        elif model.method_name == 'RIVGAE':
            embedding_list, hx, loss_data_list = model(x_list, edge_list, hx)
            loss_input_list = loss_data_list
            loss_input_list.append(adj_list)
        else:  # GCRN, EvolveGCN
            embedding_list = model(x_list, adj_list)
            loss_input_list = [embedding_list, batch_indices]
        output_list = embedding_list
        return loss_input_list, output_list, hx

    def get_batch_info(self, batch_size):
        batch_num = self.node_num // batch_size
        if self.node_num % batch_size != 0:
            batch_num += 1
        return batch_num

    # edge_list parameter is only used by VGRNN, node_dist_list parameter is only used by PGNN
    def learn_embedding(self, adj_list, x_list, edge_list=None, node_dist_list=None, epoch=50, batch_size=1024, lr=1e-3, start_idx=0, weight_decay=0., model_file='rivgae', load_model=False, shuffle=True, export=True):
        print('start learning embedding!')
        model, loss_model, optimizer, _ = self.prepare(load_model, model_file, lr=lr, weight_decay=weight_decay)
        batch_num = self.get_batch_info(batch_size)
        all_nodes = torch.arange(self.node_num, device=self.device)
        output_list = []

        st = time.time()
        print('start unsupervised training!')
        model.train()
        for i in range(epoch):
            node_indices = all_nodes[torch.randperm(self.node_num)] if shuffle else all_nodes  # Tensor
            hx = None  # used for VGRNN
            for j in range(batch_num):
                batch_indices = node_indices[j * batch_size: min(self.node_num, (j + 1) * batch_size)]
                t1 = time.time()
                loss_input_list, output_list, hx = self.get_model_res(adj_list, x_list, edge_list, node_dist_list, model, batch_indices, hx)
                loss = loss_model(loss_input_list)
                loss.backward()
                # gradient accumulation
                if j == batch_num - 1:
                    optimizer.step()  # update gradient
                    model.zero_grad()
                t2 = time.time()
                self.clear_cache()
                print('epoch', i + 1, ', batch num = ', j + 1, ', loss:', loss.item(), ', cost time: ', t2 - t1, ' seconds!')
        print('end unsupervised training!')
        en = time.time()
        cost_time = en - st

        if export:
            self.save_embedding(output_list, start_idx)
        # if model_file is None, then the model would not be saved
        if model_file:
            torch.save(model.state_dict(), os.path.join(self.model_base_path, model_file))
        del adj_list, x_list, output_list, model
        self.clear_cache()
        print('learning embedding total time: ', cost_time, ' seconds!')
        return cost_time

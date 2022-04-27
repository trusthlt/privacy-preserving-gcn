from model import GCNNet
from utils import EarlyStopping, subsample_graph, make_small_reddit, get_train_edge_count, prepare_pokec_main, random_graph_split
from privacy import DPSGD, DPAdam
from accountant import get_priv
from settings import Settings

import os
import random
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'
import matplotlib.pyplot as plt

import numpy as np
import pdb
import time

import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.nn import MessagePassing, DataParallel
from torch_geometric.nn.inits import reset, uniform
from torch_geometric.utils import add_self_loops, degree


class GCNModel(object):
    def __init__(self, ss):
        self.dataset = ss.args.dataset
        self.root_dir = ss.root_dir
        self.log_dir = ss.log_dir
        self.time_dir = ss.time_dir
        self.model_name = ss.model_name
        self.epochs = ss.args.epochs
        self.hidden_dim = ss.args.hidden_dim
        self.learning_rate = ss.args.learning_rate
        self.weight_decay = ss.args.weight_decay
        self.amsgrad = ss.args.amsgrad
        self.verbose = ss.args.verbose
        self.activation = ss.args.activation
        self.dropout = ss.args.dropout
        self.momentum = ss.args.momentum
        self.early_stopping = ss.args.early_stopping
        self.patience = ss.args.patience
        self.optim_type = ss.args.optim_type
        self.parallel = ss.args.parallel

        self.learning_rate_decay = False
        self.scheduler = None
        self.max_epochs_lr_decay = 200
        self.scheduler_gamma = 1

        self.subsample_rate = ss.args.subsample_rate
        self.split_graph = ss.args.split_graph
        self.split_n_subgraphs = ss.args.split_n_subgraphs

        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')

        self.seed = ss.args.seed

        self.total_params = 0
        self.trainable_params = 0

        # Privacy parameters
        self.private = ss.args.private
        self.delta = ss.args.delta
        self.noise_scale = ss.args.noise_scale
        self.gradient_norm_bound = ss.args.gradient_norm_bound
        self.lot_size = ss.args.lot_size
            # Number of subgraphs in a lot, if no graph splitting then 1
        self.sample_size = ss.args.sample_size
        self.alpha = None
        self.total_samples = self.split_n_subgraphs

        # Results
        self.train_accs = []
        self.train_losses = []
        self.train_f1s = []
        self.valid_accs = []
        self.valid_losses = []
        self.valid_f1s = []
        self.test_loss = None
        self.test_acc = None
        self.test_f1 = None
        self.random_baseline = True
        self.majority_baseline = True

        self.data = self.get_dataloader(dataset=self.dataset,
                                        pokec_feat_type=ss.args.pokec_feat_type,
                                        get_edge_counts=False)

        self._init_model()

    def get_dataloader(self, dataset='cora', pokec_feat_type='sbert',
                       get_edge_counts=False):
        '''
        Prepares the dataloader for a particular split of the data.
        '''
        if dataset == 'cora':
            data = Planetoid(self.root_dir, "Cora")[0]
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False

        elif dataset == 'citeseer':
            data = Planetoid(self.root_dir, "CiteSeer")[0]
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False

        elif dataset == 'pubmed':
            data = Planetoid(self.root_dir, "PubMed")[0]
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False

        elif dataset == 'reddit':
            data = Reddit(os.path.join(self.root_dir, 'Reddit'))[0]
        elif dataset == 'reddit-small':
            try:
                data = torch.load(os.path.join(self.root_dir, 'RedditS',
                                               'processed', 'data.pt'))
            except FileNotFoundError:
                print("Small reddit data not found, preparing...")
                data = make_small_reddit(rate=0.1)
        elif dataset == 'pokec-pets':
            try:
                data = torch.load(
                        os.path.join(self.root_dir, 'Pokec', 'processed',
                                     f'pokec-pets_{pokec_feat_type}_cased.pt')
                        )
            except FileNotFoundError:
                print("Pokec dataset not found, preparing...")
                data = prepare_pokec_main(feat_type=pokec_feat_type)
        else:
            raise Exception("Incorrect dataset specified.")

        ###
        # Place code here for mini-batching/graph splitting
        ###
        if self.split_graph:
            batch_masks = random_graph_split(data, n_subgraphs=self.split_n_subgraphs)
            batch_masks = [mask.to(self.device) for mask in batch_masks]
            data.batch_masks = batch_masks
            num_sample_nodes = data.x[batch_masks[0]].shape[0]
            print(f"Split graph into {self.split_n_subgraphs} subgraphs of "
                  f"{num_sample_nodes} nodes.")

        if self.subsample_rate != 1.:
            if self.split_graph:
                raise Exception("Functionality not included for subsampling \
                                graph after splitting it into sub-graphs.")
            print("Subsampling graph...")
            subsample_graph(data, rate=self.subsample_rate,
                            maintain_class_dists=True)
        print(f"Total number of nodes: {data.x.shape[0]}")
        print(f"Total number of edges: {data.edge_index.shape[1]}")
        print(f"Number of train nodes: {data.train_mask.sum().item()}")
        print(f"Number of validation nodes: {data.val_mask.sum().item()}")
        print(f"Number of test nodes: {data.test_mask.sum().item()}")

        data = data.to(self.device)
        if get_edge_counts:
            if self.split_graph:
                print("Graph split: Showing edge count for first subgraph.")
            num_train_edges, num_test_edges = get_train_edge_count(data, split_graph=self.split_graph)
            print(f"Number of train edges: {num_train_edges}")
            print(f"Number of test edges: {num_test_edges}")

        self.num_nodes = data.x.shape[0]
        self.num_edges = data.edge_index.shape[1]

        return data

    def _init_model(self):

        self.num_classes = len(torch.unique(self.data.y))
        print("Using {}".format(self.device))
        model = GCNNet(self.data.num_node_features, self.num_classes,
                       self.hidden_dim, self.device, self.activation,
                       self.dropout).double().to(self.device)

        if self.parallel:
            model = DataParallel(model)

        for param in model.parameters():
            print(param.shape)

        total_params = 0
        for param in list(model.parameters()):
            nn = 1
            for sp in list(param.size()):
                nn = nn * sp
            total_params += nn
        self.total_params = total_params
        print("Total parameters", self.total_params)

        model_params = filter(lambda param: param.requires_grad,
                              model.parameters())
        trainable_params = sum([np.prod(param.size())
                                for param in model_params])
        self.trainable_params = trainable_params
        print("Trainable parameters", self.trainable_params)

        if self.private:
            if self.optim_type == 'sgd':
                self.optimizer = DPSGD(model.parameters(), self.noise_scale,
                                       self.gradient_norm_bound, self.lot_size,
                                       self.sample_size, lr=self.learning_rate)
            elif self.optim_type == 'adam':
                self.optimizer = DPAdam(model.parameters(), self.noise_scale,
                                        self.gradient_norm_bound,
                                        self.lot_size, self.sample_size,
                                        lr=self.learning_rate)
            else:
                raise Exception(f"{self.optim_type} not a valid optimizer (adam or sgd).")
        else:
            if self.optim_type == 'sgd':
                self.optimizer = torch.optim.SGD(model.parameters(),
                                                 lr=self.learning_rate,
                                                 weight_decay=self.weight_decay)

            elif self.optim_type == 'adam':
                self.optimizer = torch.optim.Adam(model.parameters(),
                                                  lr=self.learning_rate,
                                                  weight_decay=self.weight_decay)
            else:
                raise Exception(f"{self.optim_type} not a valid optimizer (adam or sgd).")

        if self.learning_rate_decay:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1,
                                                             gamma=self.scheduler_gamma)

        self.loss = torch.nn.NLLLoss()

        self.model = model

    def train(self):
        model = self.model
        optimizer = self.optimizer
        early_stopping = EarlyStopping(self.patience)

        model.train()

        parameters = []
        q = self.lot_size / self.total_samples
            # 'Sampling ratio'
            # Number of subgraphs in a lot divided by total number of subgraphs
            # If no graph splitting, both values are 1
        max_range = self.epochs / q  # max number of Ts
        max_parameters = [(q, self.noise_scale, max_range)]

        print('Training...')
        for epoch in range(self.epochs):
            if self.split_graph:
                random.shuffle(self.data.batch_masks)

            if self.private:
                if self.split_graph:
                    batch_losses = []
                    batch_accs = []
                    batch_precs = []
                    batch_recs = []
                    batch_f1s = []
                    lot_t = 0
                    for idx in range(self.split_n_subgraphs):
                        T_k = (lot_t + 1) + ((1 / q) * epoch)
                        optimizer.zero_accum_grad()
                        optimizer.zero_sample_grad()
                        pred_prob_node = model(self.data)
                        loss = self.loss(pred_prob_node[self.data.batch_masks[idx]],
                                         self.data.y[self.data.batch_masks[idx]])
                        loss.backward()
                        optimizer.per_sample_step()
                        optimizer.step(self.device)

                        parameters = [(q, self.noise_scale, T_k)]
                        eps, delta = get_priv(parameters, delta=self.delta,
                                              max_lmbd=32)
                        maxeps, maxdelta = get_priv(max_parameters,
                                                    delta=self.delta, max_lmbd=32)
                        b_acc, b_prec, b_rec, b_f1 = self.calculate_accuracy(
                                pred_prob_node[self.data.batch_masks[idx]],
                                self.data.y[self.data.batch_masks[idx]])

                        batch_losses.append(loss.item())
                        batch_accs.append(b_acc)
                        batch_precs.append(b_prec)
                        batch_recs.append(b_rec)
                        batch_f1s.append(b_f1)

                        lot_t += 1
                        print("Spent privacy (function accountant): \n", eps)
                        print("Spent MAX privacy (function accountant): \n", maxeps)

                else:
                    lot_t = 0  # For 1-graph datasets, always batch of 1
                    T_k = (lot_t + 1) + ((1 / q) * epoch)

                    optimizer.zero_accum_grad()
                    optimizer.zero_sample_grad()
                    pred_prob_node = model(self.data)
                    loss = self.loss(pred_prob_node[self.data.train_mask],
                                     self.data.y[self.data.train_mask])
                    loss.backward()
                    optimizer.per_sample_step()
                    optimizer.step(self.device)

                    parameters = [(q, self.noise_scale, T_k)]
                    eps, delta = get_priv(parameters, delta=self.delta,
                                          max_lmbd=32)
                    maxeps, maxdelta = get_priv(max_parameters,
                                                delta=self.delta, max_lmbd=32)
                    print("Spent privacy (function accountant): \n", eps)
                    print("Spent MAX privacy (function accountant): \n", maxeps)
            else:
                if self.split_graph:
                    batch_losses = []
                    batch_accs = []
                    batch_precs = []
                    batch_recs = []
                    batch_f1s = []
                    for idx in range(self.split_n_subgraphs):
                        optimizer.zero_grad()
                        pred_prob_node = model(self.data)
                        loss = self.loss(pred_prob_node[self.data.batch_masks[idx]],
                                         self.data.y[self.data.batch_masks[idx]])
                        loss.backward()
                        optimizer.step()

                        b_acc, b_prec, b_rec, b_f1 = self.calculate_accuracy(
                                pred_prob_node[self.data.batch_masks[idx]],
                                self.data.y[self.data.batch_masks[idx]])

                        batch_losses.append(loss.item())
                        batch_accs.append(b_acc)
                        batch_precs.append(b_prec)
                        batch_recs.append(b_rec)
                        batch_f1s.append(b_f1)
                else:
                    optimizer.zero_grad()
                    pred_prob_node = model(self.data)
                    loss = self.loss(pred_prob_node[self.data.train_mask],
                                     self.data.y[self.data.train_mask])
                    loss.backward()
                    optimizer.step()

            if self.scheduler != None and epoch < self.max_epochs_lr_decay:
                print("Old LR:", self.optimizer.param_groups[0]['lr'])
                self.scheduler.step()
                print("New LR:", self.optimizer.param_groups[0]['lr'])

            if self.split_graph:
                loss = np.mean(batch_losses)
                accuracy = np.mean(batch_accs)
                prec = np.mean(batch_precs)
                rec = np.mean(batch_recs)
                f1 = np.mean(batch_f1s)
            else:
                accuracy, prec, rec, f1 = self.calculate_accuracy(
                        pred_prob_node[self.data.train_mask],
                        self.data.y[self.data.train_mask])
                loss = loss.item()

            self.log(epoch, loss, accuracy, prec, rec, f1,
                     split='train')
            self.train_losses.append(loss)
            self.train_accs.append(accuracy)
            self.train_f1s.append(f1)

            val_loss = self.evaluate_on_valid(model, epoch)

            self.plot_learning_curve()
            if self.early_stopping:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    break
            print('\n')

        if self.early_stopping:
            return -early_stopping.best_score
        else:
            return val_loss

    def calculate_accuracy(self, pred, target, rand_maj_baseline=False):

        if rand_maj_baseline:
            pred_node = pred
        else:
            _, pred_node = pred.max(dim=1)
        acc = float(pred_node.eq(target).sum().item()) / (len(pred_node))

        results = precision_recall_fscore_support(
                target.cpu().numpy(), pred_node.cpu().numpy(), average='micro')
        prec, rec, f1, _ = results

        return acc, prec, rec, f1

    def log(self, epoch, loss, accuracy, prec, rec, f1, split='val'):

        if self.verbose:
            print("Epoch {} ({})\tLoss: {:.4f}\tA: {:.4f}\tP: {:.4f}\t"
                  "R: {:.4f}\tF1: {:.4f}".format(epoch, split, loss, accuracy,
                                                 prec, rec, f1), flush=True)

    def plot_learning_curve(self):
        '''
        Result png figures are saved in the log directory.
        '''
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        privacy = 'N/A'
        fig.suptitle('Model Learning Curve ({}, % data {}, epsilon {})'.format(
            self.dataset, self.subsample_rate, privacy))

        epochs = list(range(len(self.train_losses)))
        ax1.plot(epochs, self.train_losses, 'o-', markersize=2, color='b',
                 label='Train')
        ax1.plot(epochs, self.valid_losses, 'o-', markersize=2, color='c',
                 label='Validation')
        ax1.set(ylabel='Loss')

        ax2.plot(epochs, self.train_accs, 'o-', markersize=2, color='b',
                 label='Train')
        ax2.plot(epochs, self.valid_accs, 'o-', markersize=2, color='c',
                 label='Validation')
        ax2.set(xlabel='Epoch', ylabel='Accuracy')
        ax1.legend()

        plt.savefig(os.path.join(self.time_dir, 'learning_curve.png'))
        plt.close()

    def evaluate_on_valid(self, model, epoch, early_stopping=None):

        model.eval()

        with torch.no_grad():
            pred_prob_node = model(self.data)
            loss = self.loss(pred_prob_node[self.data.val_mask],
                             self.data.y[self.data.val_mask])

            accuracy, prec, rec, f1 = self.calculate_accuracy(
                    pred_prob_node[self.data.val_mask],
                    self.data.y[self.data.val_mask])

            self.log(epoch, loss, accuracy, prec, rec, f1, split='val')
            self.valid_losses.append(loss.item())
            self.valid_accs.append(accuracy)
            self.valid_f1s.append(f1)
        return loss.item()

    def evaluate_on_test(self):

        model = self.model
        model.eval()
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes,
                                       dtype=torch.long)

        test_size = self.data.y[self.data.test_mask].shape
        if self.random_baseline:
            rand_preds = torch.randint(0, self.data.y.unique().max().item(), (test_size[0],)).to(self.device)
            accuracy_rand, prec_rand, rec_rand, f1_rand = self.calculate_accuracy(
                    rand_preds,
                    self.data.y[self.data.test_mask], rand_maj_baseline=True)
            print(f"Random baseline results (test F1): {f1_rand}")
        if self.majority_baseline:
            majority = self.data.y[self.data.train_mask].bincount().argmax().item()
            majority_preds = torch.ones(test_size, device=self.device) * majority
            accuracy_maj, prec_maj, rec_maj, f1_maj = self.calculate_accuracy(
                    majority_preds,
                    self.data.y[self.data.test_mask], rand_maj_baseline=True)
            print(f"Majority baseline results (test F1): {f1_maj}")

        with torch.no_grad():
            pred_prob_node = model(self.data)
            loss = self.loss(pred_prob_node[self.data.test_mask],
                             self.data.y[self.data.test_mask])
            preds = pred_prob_node.max(dim=1)[1]

            accuracy, prec, rec, f1 = self.calculate_accuracy(
                    pred_prob_node[self.data.test_mask],
                    self.data.y[self.data.test_mask])

            for t, p in zip(self.data.y[self.data.test_mask],
                            preds[self.data.test_mask]):
                if p.long() in range(self.num_classes):
                    confusion_matrix[t.long(), p.long()] += 1
                else:
                    confusion_matrix[t.long(), -1] += 1

            confusion_out = confusion_matrix.data.cpu().numpy()
            np.savetxt(os.path.join(self.time_dir, 'confusion_matrix.csv'),
                       confusion_out, delimiter=',', fmt='% 4d')

            # Output predictions
            print("Preparing predictions file...\n")
            pred_filename = os.path.join(self.time_dir,
                                         f'preds_seed{self.seed}.csv')
            with open(pred_filename, 'w') as pred_f:
                pred_f.write("Pred,Y\n")
                for idx in range(preds[self.data.test_mask].shape[0]):
                    pred_f.write(f"{preds[self.data.test_mask][idx]},{self.data.y[self.data.test_mask][idx]}\n")

            self.test_loss = loss.item()
            self.test_acc = accuracy
            self.test_f1 = f1

            print("Test set results\tLoss: {:.4f}\tNode Accuracy: {:.4f}\t"
                  "Precision: {:.4f}\tRecall: {:.4f}\tF1: {:.4f}".format(
                      loss.item(), accuracy, prec, rec, f1))
        return loss.item(), accuracy, prec, rec, f1

    def output_results(self, best_score):
        '''
        Adds final test results to a csv file.
        '''
        filepath = os.path.join(self.time_dir, 'results.csv')
        best_val_loss = best_score
        epoch = self.valid_losses.index(best_val_loss)
        best_val_acc = self.valid_accs[epoch]
        best_val_f1 = self.valid_f1s[epoch]

        with open(filepath, 'w') as out_f:
            out_f.write('BestValidLoss,BestValidAcc,BestValidF1,'
                        'BestValidEpoch,TestLoss,TestAcc,TestF1,'
                        'NumTrainableParams,NumNodes(per_sg),NumEdges(per_sg),ModelConfig\n')
            out_f.write(f'{best_val_loss:.4f},{best_val_acc:.4f},'
                        f'{best_val_f1:.4f},{epoch},{self.test_loss:.4f},'
                        f'{self.test_acc:.4f},{self.test_f1:.4f},'
                        f'{self.trainable_params},{self.num_nodes},'
                        f'{self.num_edges},'
                        f'{self.model_name}\n')


def main():
    now = time.time()

    # Setting the seed
    ss = Settings()
    ss.make_dirs()
    torch.manual_seed(ss.args.seed)
    torch.cuda.manual_seed(ss.args.seed)
    np.random.seed(ss.args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    model = GCNModel(ss)

    best_score = model.train()
    test_loss, test_acc, test_prec, test_rec, test_f1 = model.evaluate_on_test()
    model.output_results(best_score)
    print(f"Test score: {test_loss:.4f} with accuracy {test_acc:.4f} and f1 {test_f1:.4f}")
    with open('adam_hyperparams.csv', 'a') as f:
        f.write(f"{ss.args.dataset},{ss.args.noise_scale},{ss.args.learning_rate},{test_f1:.4f}\n")

    then = time.time()
    runtime = then - now
    print(f"--- Script completed in {runtime} seconds ---")

if __name__ == '__main__':
    main()

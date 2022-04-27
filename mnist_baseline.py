import pdb
import glob
import os
import sys
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.nn import MessagePassing, DataParallel
from torch_geometric.nn.inits import reset, uniform
from torch_geometric.utils import add_self_loops, degree
from torchvision import datasets, transforms

from model import GCNNet, Conv, MLPBaseline
from dataset import GCNDataset
from utils import EarlyStopping
from privacy import DPSGD, get_priv
from settings import Settings


class BaselineModel(object):
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
        self.parallel = ss.args.parallel

        self.subsample_rate = ss.args.subsample_rate

        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')

        self.total_params = 0
        self.trainable_params = 0

        # Privacy parameters
        self.private = ss.args.private
        self.delta = ss.args.delta
        self.noise_scale = ss.args.noise_scale
        self.gradient_norm_bound = ss.args.gradient_norm_bound
        self.lot_size = ss.args.lot_size
        self.alpha = None
        self.per_sample = False
        if self.per_sample:
            self.sample_size = ss.args.sample_size
        else:
            self.sample_size = self.lot_size
        self.total_samples = 60000

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

        self.trainset, self.testset = self.get_dataloader(dataset=self.dataset)

        self._init_model()

    def get_dataloader(self, dataset='mnist'):
        '''
        Prepares the dataloader for a particular split of the data.
        '''
        transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        mnist_train = datasets.MNIST(self.root_dir, train=True, download=True,
                                     transform=transform)
        mnist_test = datasets.MNIST(self.root_dir, train=False,
                                     transform=transform)
        train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=self.lot_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=self.lot_size,
                                                  shuffle=True)

        return train_loader, test_loader

    def _init_model(self):

        print("Using {}".format(self.device))
        model = MLPBaseline(28*28, 512, 10).double().to(self.device)

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
            self.optimizer = DPSGD(model.parameters(), self.noise_scale,
                                   self.gradient_norm_bound, self.lot_size,
                                   self.sample_size, lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(),
                                             lr=self.learning_rate,
                                             weight_decay=self.weight_decay)
        self.loss = torch.nn.NLLLoss()

        self.model = model

    def train(self):
        model = self.model
        optimizer = self.optimizer
        early_stopping = EarlyStopping(self.patience)

        model.train()
        parameters = []
        q = self.lot_size / self.total_samples
        max_range = int(self.epochs / q)  # max number of Ts
        max_parameters = [(q, self.noise_scale, max_range)]

        print('Training...')
        for epoch in range(self.epochs):
            train_loss = 0
            train_acc = 0
            train_f1 = 0
            for batch, (data, target) in tqdm(enumerate(self.trainset)):
                T_k = (batch + 1) + ((1 / q) * epoch)
                data, target = data.to(self.device), target.to(self.device)

                if self.private:
                    outputs = []
                    optimizer.zero_accum_grad()
                    if self.per_sample:
                        for sample_x, sample_y in zip(data, target):
                            optimizer.zero_sample_grad()
                            output = model(sample_x)
                            loss = self.loss(output, sample_y.unsqueeze(0))
                            outputs.append(output)
                            loss.backward()
                            optimizer.per_sample_step()
                        optimizer.step(self.device)
                        output = torch.stack(outputs).squeeze(1)
                    else:
                        optimizer.zero_sample_grad()
                        output = model(data)
                        loss = self.loss(output, target)
                        outputs.append(output)
                        loss.backward()
                        optimizer.per_sample_step()
                        optimizer.step(self.device)

                    print(f"\n\nEPOCH: {epoch+1}")
                    parameters = [(q, self.noise_scale, T_k)]
                    eps, delta = get_priv(parameters, delta=self.delta, max_lmbd=32)
                    maxeps, maxdelta = get_priv(max_parameters, delta=self.delta, max_lmbd=32)
                    print("Spent privacy (function accountant): \n", eps)
                    print("Spent MAX privacy (function accountant): \n", maxeps)
                else:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = self.loss(output, target)
                    loss.backward()
                    optimizer.step()

                accuracy, prec, rec, f1 = self.calculate_accuracy(output,
                                                                  target)
                train_loss += loss.item()
                train_acc += accuracy
                train_f1 += f1

            full_train_loss = train_loss / len(self.trainset)
            full_train_acc = train_acc / len(self.trainset)
            full_train_f1 = train_f1 / len(self.trainset)
            self.log(epoch, full_train_loss, full_train_acc, 0, 0,
                     full_train_f1, split='train')
            self.train_losses.append(train_loss / len(self.trainset))
            self.train_accs.append(train_acc / len(self.trainset))
            self.train_f1s.append(train_f1 / len(self.trainset))

            val_loss = self.evaluate_on_valid(model, epoch)

            if self.early_stopping:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    break
            print('\n')

        if self.early_stopping:
            return -early_stopping.best_score
        else:
            return val_loss

    def calculate_accuracy(self, pred, target):

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
                                                 prec, rec, f1))

    def plot_learning_curve(self):
        '''
        Result png figures are saved in the log directory.
        '''
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        privacy = self.epsilon if self.private else 'N/A'
        fig.suptitle('Model Learning Curve ({}, epsilon {})'.format(
            self.dataset, privacy))

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
        test_losses = []
        test_accs = []
        test_f1s = []

        with torch.no_grad():
            for data, target in self.testset:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = F.nll_loss(output, target)

                accuracy, prec, rec, f1 = self.calculate_accuracy(output,
                                                                  target)

                test_losses.append(loss.item())
                test_accs.append(accuracy)
                test_f1s.append(f1)

            self.log(epoch, np.mean(test_losses), np.mean(test_accs), 0, 0,
                     np.mean(test_f1s), split='val')
            self.valid_losses.append(np.mean(test_losses))
            self.valid_accs.append(np.mean(test_accs))
            self.valid_f1s.append(np.mean(test_f1s))

        return loss.item()

    def evaluate_on_test(self):

        model = self.model
        model.eval()
        test_losses = []
        test_accs = []
        test_f1s = []

        with torch.no_grad():
            for data, target in self.testset:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = F.nll_loss(output, target)

                accuracy, prec, rec, f1 = self.calculate_accuracy(output,
                                                                  target)

                test_losses.append(loss.item())
                test_accs.append(accuracy)
                test_f1s.append(f1)

            print("Test set results\tLoss: {:.4f}\tNode Accuracy: {:.4f}\t"
                  "F1: {:.4f}".format(
                      np.mean(test_losses), np.mean(test_accs),
                      np.mean(test_f1s)))

        return np.mean(test_losses), np.mean(test_accs), np.mean(test_f1s)

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
                        'NumTrainableParams\n')
            out_f.write(f'{best_val_loss:.4f},{best_val_acc:.4f},'
                        f'{best_val_f1:.4f},{epoch},{self.test_loss:.4f},'
                        f'{self.test_acc:.4f},{self.test_f1:.4f},'
                        f'{self.trainable_params}\n')


def main():
    # Setting the seed
    ss = Settings()
    torch.manual_seed(ss.args.seed)
    torch.cuda.manual_seed(ss.args.seed)
    np.random.seed(ss.args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    model = BaselineModel(ss)

    best_score = model.train()
    test_loss, test_acc, test_f1 = model.evaluate_on_test()
    print(f"Test score: {test_loss:.4f} with accuracy {test_acc:.4f} and f1 {test_f1:.4f}")


if __name__ == '__main__':
    main()

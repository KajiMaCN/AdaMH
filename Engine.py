import torch
import numpy as np

from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from model.AdaMH import AdaMH, IndexGenerator
from model.Dataset import DataSet


class Engine:
    def __init__(self, num_circrna, num_disease, args):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.patience = args.patience
        self.feats_type = args.feats_type
        self.num_ntype = args.num_ntype
        self.num_circrna = num_circrna
        self.num_disease = num_disease
        self.adj = np.zeros([self.num_circrna, self.num_disease])
        self.etypes_lists = [
            [[0, 1], [2, 3]],
            [[1, 0], [4, 5], [None]]
        ]
        self.expected_metapaths = [
            [(0, 1, 0), (0, 2, 0)],
            [(1, 0, 1), (1, 2, 1), (1, 1)]
        ]
        self.mask_type = {
            "use_mask": [[True, False], [True, False, False]],
            "no_mask": [[False] * 2, [False] * 3]
        }
        self.num_epochs = args.epochs
        self.batch_size = args.batch_size
        self.loss_his = []
        self.esp = 0.5

    def init_model(self, args):
        self.datasets = DataSet()
        self.datasets.create_feature(self.feats_type, self.num_ntype, self.device)
        self.ada_mh = AdaMH(
            self.datasets,
            self.num_circrna,
            self.mask_type,
            self.etypes_lists,
            self.device,
            args
        )
        self.optimizer = torch.optim.Adam(
            self.ada_mh.magnn_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def prepare_data(self):
        self.train_pos = self.datasets.pos_data['train_pos_circrna_disease']
        self.val_pos = self.datasets.pos_data['val_pos_circrna_disease']
        self.test_pos = self.datasets.pos_data['test_pos_circrna_disease']
        self.train_neg = self.datasets.neg_data['train_neg_circrna_disease']
        self.val_neg = self.datasets.neg_data['val_neg_circrna_disease']
        self.test_neg = self.datasets.neg_data['test_neg_circrna_disease']
        self.y_true_test = np.array([1] * len(self.test_pos) + [0] * len(self.test_neg))

        for i in range(len(self.train_pos[0])):
            self.adj[self.train_pos[i][0], self.train_pos[i][1]] = 1

    def calc_reward(self, last_losses, eps):
        if len(last_losses) < 3:
            return 1.0
        cur_decrease = last_losses[-2] - last_losses[-1]
        avg_decrease = np.mean([last_losses[i] - last_losses[i + 1] for i in range(len(last_losses) - 2)])
        return 1 if cur_decrease > avg_decrease else eps

    def modify_model(self, args):
        self.ada_mh = AdaMH(
            self.datasets,
            self.num_circrna,
            self.mask_type,
            self.etypes_lists,
            self.device,
            args
        )

    def train_iteration(self, train_pos_batch, train_neg_batch):
        x_pos, x_neg, train_loss, cl_loss = self.ada_mh(train_pos_batch, train_neg_batch)
        neg_score = torch.sum(x_neg, -1) + x_pos
        loss_gp = torch.sum(x_pos / (neg_score + 1e-8) + 1e-8)
        self.loss_his.append(train_loss.item())
        reward = self.calc_reward(self.loss_his, self.esp)
        loss = train_loss * reward + 0.001 * loss_gp + 0.001 * cl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, epoch_info):
        train_pos_idx_generator = IndexGenerator(batch_size=self.batch_size, num_data=len(self.train_pos))
        self.ada_mh.train()
        for iteration in range(train_pos_idx_generator.num_iterations()):
            train_pos_idx_batch = train_pos_idx_generator.next()
            train_pos_batch = self.train_pos[train_pos_idx_batch].tolist()
            train_neg_idx_batch = np.random.choice(len(self.train_neg), len(train_pos_idx_batch))
            train_neg_batch = self.train_neg[train_neg_idx_batch].tolist()
            loss_value = self.train_iteration(train_pos_batch, train_neg_batch)
            if loss_value is None:
                continue
            print("\r", f'{epoch_info} | Iteration {iteration:05d} | Train_Loss {loss_value:.4f} ', end="", flush=True)

    def test_model(self):
        test_idx_generator = IndexGenerator(batch_size=self.batch_size, num_data=len(self.test_pos), shuffle=False)
        self.ada_mh.eval()
        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            for _ in range(test_idx_generator.num_iterations()):
                test_idx_batch = test_idx_generator.next()
                test_pos_batch = self.test_pos[test_idx_batch].tolist()
                test_neg_batch = self.test_neg[test_idx_batch].tolist()

                x_pos, x_neg, _, _ = self.ada_mh(test_pos_batch, test_neg_batch)

                pos_proba_list.append(x_pos)
                neg_proba_list.append(x_neg)
        y_proba_test = torch.cat(pos_proba_list + neg_proba_list).cpu().numpy()
        return y_proba_test

    def calculate_metrics(self, y_proba_test):
        auc = roc_auc_score(self.y_true_test, y_proba_test)
        precision, recall, _ = precision_recall_curve(self.y_true_test, y_proba_test)
        aupr = metrics.auc(recall, precision)
        ap = average_precision_score(self.y_true_test, y_proba_test)
        pred_label = (y_proba_test > 0.8).astype(int)
        acc = accuracy_score(self.y_true_test, pred_label)
        f1 = f1_score(self.y_true_test, pred_label)
        mcc = matthews_corrcoef(self.y_true_test, pred_label)
        p = precision_score(self.y_true_test, pred_label)
        recall_s = recall_score(self.y_true_test, pred_label)
        fpr, tpr, _ = metrics.roc_curve(self.y_true_test, y_proba_test)
        return {
            'auc': auc,
            'aupr': aupr,
            'ap': ap,
            'acc': acc,
            'f1': f1,
            'mcc': mcc,
            'precision': p,
            'recall': recall_s,
            'fpr': fpr,
            'tpr': tpr
        }

    def run(self, repeats):
        all_metrics = []

        for repeat in range(repeats):
            for epoch in range(self.num_epochs):
                epoch_info = f'repeat {repeat:05d} | Epoch {epoch:05d}'
                self.train_epoch(epoch_info)
            y_proba_test = self.test_model()
            metrics = self.calculate_metrics(y_proba_test)
            all_metrics.append(metrics)

        mean_metrics = {
            key: np.mean([m[key] for m in all_metrics]) for key in
            ['auc', 'aupr', 'f1', 'mcc', 'precision', 'recall', 'acc']
        }

        print(f"mean_auc: {mean_metrics['auc']} mean_aupr:{mean_metrics['aupr']} | mean_F1-score:{mean_metrics['f1']} "
              f"| mean_MCC:{mean_metrics['mcc']} | mean_Precision:{mean_metrics['precision']} | "
              f"mean_RECALL:{mean_metrics['recall']} | mean_acc:{mean_metrics['acc']}")

        return mean_metrics['auc']

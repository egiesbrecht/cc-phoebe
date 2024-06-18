from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from json import JSONEncoder
import numpy as np
import datetime


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    """
    http://stackoverflow.com/q/32239577/395857
    """
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) /   float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


class EpochStats(JSONEncoder):
    def __init__(self, name, id2label, chomsky_task=False):
        self.name = name
        self.chomsky_task = chomsky_task
        self.id2label = id2label
        self.scores = {}
        self.scores_by_label = {}

    def add_score(self, metric_type, score):
        if metric_type not in self.scores:
            self.scores[metric_type] = []
        self.scores[metric_type].append(score)

    def add_label_score(self, metric_type, labels, score):
        if metric_type not in self.scores_by_label:
            self.scores_by_label[metric_type] = {}
        for l in labels:
            if l not in self.scores_by_label:
                self.scores_by_label[metric_type][l] = []
            if isinstance(score, Iterable):
                for c in score:
                    self.scores_by_label[metric_type][l].append(c)
            else:
                self.scores_by_label[metric_type][l].append(score)

    def last_scores(self):
        res = {}
        for k, v in self.scores.items():
            res[k] = v[-1]
        return res

    def calc_avg_scores(self):
        self.avg_scores = {}
        for k, v in self.scores.items():
            self.avg_scores[k] = sum(v) / len(v)
        return self.avg_scores

    def calc_avg_scores_by_label(self):
        self.avg_scores_by_label = {}
        for k, t in self.scores_by_label.items():
            self.avg_scores_by_label[k] = {}
            for l, v in t.items():
                l = self.id2label[l]
                self.avg_scores_by_label[k][l] = sum(v) / len(v)
        return self.avg_scores_by_label

    def default(self, o):
        return o.__dict__

    def flat_metrics(self, preds, labels, calc_scores_by_label=False, zd_val=1, one_label_only=True):
        #print(preds.shape, preds)
        #print(labels.shape, labels)
        if self.chomsky_task:
            y_pred = preds
        elif one_label_only:
            y_pred = np.array([np.argmax(n) for n in preds])
        else:
            l_one_idxs = [np.where(n == 1)[0].tolist() for n in labels]
            num_labels = [len(n) for n in l_one_idxs]
            inds = [(np.argpartition(p, -n)[-n:].tolist(), len(p)) for p, n in zip(preds, num_labels)]
            y_pred = np.array([[1 if i in ns else 0 for i in range(l)] for ns, l in inds], dtype=int)
        if calc_scores_by_label:
            for yt, yp, lp in zip(labels, y_pred, l_one_idxs):
                yp_n = [yp]
                yt_n = [yt]
                acc = accuracy_score(yt_n, yp_n)
                self.add_label_score("accuracy", lp, acc)
                if len(labels) > 2:
                    ham = hamming_score(yt_n, yp_n)
                    self.add_label_score("hamming", lp, ham)
                f1 = f1_score(yt_n, yp_n, average=None, zero_division=zd_val)
                self.add_label_score("f1", lp, f1)
                rec = recall_score(yt_n, yp_n, average=None, zero_division=zd_val)
                self.add_label_score("recall", lp, rec)
                prec = precision_score(yt_n, yp_n, average=None, zero_division=zd_val)
                self.add_label_score("precision", lp, rec)
        #print(y_pred.shape, y_pred)
        #print(labels.shape, labels)
        
        #y_pred = (y_pred == 0)#.int()

        if self.chomsky_task:
            self.add_score("accuracy", (y_pred == labels).mean())
        else:
            self.add_score("accuracy", accuracy_score(labels, y_pred))
            if len(labels) > 2:
                self.add_score("hamming", hamming_score(labels, y_pred))
            self.add_score("per class f1", pc_f1(y_pred, labels, False, False))
            self.add_score("f1_micro", f1_score(labels, y_pred, average="micro", zero_division=zd_val))
            self.add_score("f1_macro", f1_score(labels, y_pred, average="macro", zero_division=zd_val))
            self.add_score("recall_micro", recall_score(labels, y_pred, average="micro", zero_division=zd_val))
            self.add_score("recall_macro", recall_score(labels, y_pred, average="macro", zero_division=zd_val))
            self.add_score("precision_micro", precision_score(labels, y_pred, average="micro", zero_division=zd_val))
            self.add_score("precision_macro", precision_score(labels, y_pred, average="macro", zero_division=zd_val))


def flat_metrics(preds, labels, stats: EpochStats, calc_scores_by_label=False, zd_val=1):
    stats.flat_metrics(preds, labels, calc_scores_by_label, zd_val)


def print_stat_tuples(stats):
    def print_scores(s):
        for k, v in s.calc_avg_scores().items():
            print(f"    {k}: {v}")

    assert len(stats) == 2, len(stats)
    for i, (r, e) in enumerate(zip(*stats)):
        print(f"Epoch {i+1}:")
        print(f"  Train:")
        print_scores(r)
        print(f"  Test:")
        print_scores(e)
        print()


def pc_f1(y_pred, y_true, print_statistics=True, convert_predictions=True):
    num_classes = y_true.shape[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        if convert_predictions:
            one_hot_pred = np.zeros((len(y_pred), num_classes))
            one_hot_pred[np.arange(len(y_pred)), y_pred] = 1

            #one_hot_gt = np.zeros((len(y_true), num_classes))
            #one_hot_gt[np.arange(len(y_true)), y_true] = 1
            one_hot_gt = y_true
        else:
            one_hot_pred = y_pred
            one_hot_gt = y_true

        actually_there = (np.sum(one_hot_gt, axis=0) > 0).astype("int32")

        tp = np.sum(one_hot_pred * one_hot_gt, axis=0)
        fp = np.sum(one_hot_pred * (1-one_hot_gt), axis=0)
        tn = np.sum((1-one_hot_pred) * (1-one_hot_gt), axis=0)
        fn = np.sum((1-one_hot_pred) * one_hot_gt, axis=0)

        precision = tp / (tp + fp)
        recall = tp / (fn + tp)
        f1 = 2 * precision * recall / (precision + recall)
        precision[np.isnan(precision)] = 0
        recall[np.isnan(recall)] = 0
        f1[np.isnan(f1)] = 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        if print_statistics:
            print("\nPer class metrics:")
            print("  Class | Precision  |  Recall  |    F1   |  Accuracy")
            for c in range(num_classes):
                if actually_there[c] > 0:
                    print(f"   {c:2d}   |   {precision[c]:.3f}    |  {recall[c]:.3f}   |  {f1[c]:.3f}  |   {accuracy[c]:.3f}")
                else:
                    print(f"    -   |     -      |    -     |    -    |     -   ")

        macro_f1 = np.sum(f1*actually_there) / np.sum(actually_there)
        if print_statistics:
            print(f"Macro F1: {macro_f1}")

        micro_precision = np.mean(np.sum(one_hot_gt * one_hot_pred, axis=1) / np.sum(one_hot_pred, axis=1))
        micro_recall = np.mean(np.sum(one_hot_gt * one_hot_pred, axis=1) / np.sum(one_hot_gt, axis=1))
        micro_f1 = 2 * micro_recall * micro_precision / (micro_recall + micro_precision)
    return macro_f1
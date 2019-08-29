from sklearn.linear_model import LogisticRegression
from data_loader import Data_loader
import random
import numpy as np
from scipy import sparse
import pickle as pkl
from sklearn.metrics import f1_score, classification_report
from util import trim


class Logistic_regr():
    def __init__(self, mode):
        assert mode in ['train', 'eval']

        if mode == 'train':
            dl = Data_loader(labeled_only=True, option='both')
            self.train_test_val_data = dl.cv_data(0)[0] + dl.cv_data(0)[1] + dl.cv_data(0)[2]
            self.train()
        else:
            model_dict = pkl.load(open("../data/logistic_regression.pkl", 'rb'))
            self.thresholds = model_dict['thresholds']
            self.classifiers = model_dict['models']


    def train(self):
        random.shuffle(self.train_test_val_data)
        train_data = self.train_test_val_data[:3158]
        val_data = self.train_test_val_data[3158: 3158 + 790]
        test_data = self.train_test_val_data[3158 + 790:]

        train_labels = self.create_classidx_label(train_data)
        val_labels = self.create_classidx_label(val_data)
        test_labels = self.create_classidx_label(test_data)

        train_features = self.create_vectorized_representation(train_data)
        val_features = self.create_vectorized_representation(val_data)
        test_features = self.create_vectorized_representation(test_data)

        classifiers = []
        best_threshold = []
        for class_idx in range(2):
            true_train_labels = (train_labels == class_idx)
            classifier = LogisticRegression()
            classifier.fit(train_features, true_train_labels)
            classifiers.append(classifier)

            #tune threshold
            val_pred_score = classifier.predict_proba(val_features)[:, 1]
            true_val_labels = (val_labels == class_idx)
            best_t, best_f_val = 0, -1
            for t in np.arange(0.01, 1, 0.01):
                lr_label = [1 if lr_s > t else 0 for lr_s in val_pred_score]
                f_val = f1_score(true_val_labels, lr_label)
                if f_val > best_f_val:
                    best_f_val = f_val
                    best_t = t
            best_threshold.append(best_t)

        model_dict = {}
        model_dict['thresholds'] = best_threshold
        model_dict['models'] = classifiers
        self.thresholds, self.classifiers = best_threshold, classifiers
        pkl.dump(model_dict, open('../data/logistic_regression.pkl', 'wb'))

        #evaluate macro test performance (0, 1, 2 as label)
        agg_pred_test_scores = classifiers[0].predict_proba(test_features)[:, 1]
        loss_pred_test_scores = classifiers[1].predict_proba(test_features)[:, 1]
        pred_test_labels = []
        for i in range(len(agg_pred_test_scores)):
            if agg_pred_test_scores[i] >= best_threshold[0]:
                pred_test_labels.append(0)
            elif loss_pred_test_scores[i] >= best_threshold[1]:
                pred_test_labels.append(1)
            else:
                pred_test_labels.append(2)
        print('test evaluate f1-score: ', f1_score(test_labels, pred_test_labels, average='macro'))
        print(classification_report(test_labels, pred_test_labels))


    def create_classidx_label(self, tweet_data):
        """
        Extract the labels Aggression, Loss, Other to 0, 1, 2.
        """
        labels = []
        for idx in range(len(tweet_data)):
            if tweet_data[idx]['label'] == 'Aggression':
                labels.append(0)
            elif tweet_data[idx]['label'] == 'Loss':
                labels.append(1)
            else:
                labels.append(2)
        return np.array(labels)


    def create_vectorized_representation(self, tweet_data):
        """
        Return the count vectorized and tf-transformed representation of the input tweets.
        """
        sentences = [trim(tweet['word_padded_int_arr']) for tweet in tweet_data]
        features = []
        for sentence in sentences:
            representation = np.zeros((40000,))
            for word_id in sentence:
                representation[word_id] += 1
            features.append(representation)
        features = sparse.csr_matrix(features)
        return features


    def predict(self, word_int_arrs, mode='score'):
        """
        Predict probability/label of aggression/loss of batch of word int arrs.
        """
        sentences = [trim(x) for x in word_int_arrs]
        features = []
        for sentence in sentences:
            representation = np.zeros((40000,))
            for word_id in sentence:
                representation[word_id] += 1
            features.append(representation)
        sparse_features = sparse.csr_matrix(features)
        aggression_pred_scores = self.classifiers[0].predict_proba(sparse_features)[:, 1]
        loss_pred_scores = self.classifiers[1].predict_proba(sparse_features)[:, 1]

        assert mode in ['score', 'binary']
        if mode == 'score':
            return aggression_pred_scores, loss_pred_scores
        else:
            aggression_pred_labels = [1 if x >= self.thresholds[0] else 0 for x in aggression_pred_scores]
            loss_pred_labels = [1 if x >= self.thresholds[1] else 0 for x in loss_pred_scores]
            return aggression_pred_labels, loss_pred_labels


if __name__ == '__main__':
    #lr = Logistic_regr(mode='train')
    lr = Logistic_regr(mode='eval')
    #print(lr.predict([[458, 32, 1109, 994, 509, 952, 0, 0, 0, 0, 0, 0]], mode='score'))
    #print(lr.predict([[458, 32, 9, 1109, 994, 509, 952, 0, 0, 0, 0, 0, 0]], mode='score'))
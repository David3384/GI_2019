import numpy as np
from sklearn.metrics import f1_score
import pandas as pd

from model_info import model_ids

def evaluate_experiment_result(experiment_dir):
    fold_result = np.loadtxt(experiment_dir + '/result_by_fold.np')
    fold_result = np.reshape(fold_result, (-1, 4, 3))  # shape (5, 4, 3), 5 folds, 4 metrics, 3 categories
    f1_score = fold_result[:, 2, :]
    avg_agg_f1, avg_loss_f1, avg_other_f1 = np.mean(f1_score, axis=0)
    return avg_agg_f1, avg_loss_f1, avg_other_f1


def evaluate_model_result(model_experiment_dir_root):
    agg_f1_scores, loss_f1_scores, other_f1_scores = [], [], []
    for run_idx in range(5):
        experiment_dir = model_experiment_dir_root + str(run_idx)
        avg_agg_f1, avg_loss_f1, avg_other_f1 = evaluate_experiment_result(experiment_dir)
        agg_f1_scores.append(avg_agg_f1)
        loss_f1_scores.append(avg_loss_f1)
        other_f1_scores.append(avg_other_f1)
    model_scores = {}
    model_scores['agg_f1'] = sum(agg_f1_scores) / len(agg_f1_scores)
    model_scores['loss_f1'] = sum(loss_f1_scores) / len(loss_f1_scores)
    model_scores['other_f1'] = sum(other_f1_scores) / len(other_f1_scores)
    model_scores['macro_f1'] = np.mean(np.array([model_scores['agg_f1'], model_scores['loss_f1'], model_scores['other_f1']]))

    #calculate majority vote agg, loss, other, macro f1-score
    majority_vote_agg_f1 = []
    majority_vote_loss_f1 = []
    majority_vote_other_f1 = []
    majority_vote_macro_f1 = []

    for fold_idx in range(5):
        preds = []
        for run_idx in range(5):
            preds.append(np.loadtxt(model_experiment_dir_root + str(run_idx) + '/fold_%d_pred_test.np' % fold_idx))
        majority_vote_preds = majority_vote(preds)
        true_labels = np.loadtxt(model_experiment_dir_root + "0/fold_%d_truth_test.np" % fold_idx)
        agg_f1, loss_f1, other_f1 = f1_score(true_labels, majority_vote_preds, average=None)
        majority_vote_agg_f1.append(agg_f1)
        majority_vote_loss_f1.append(loss_f1)
        majority_vote_other_f1.append(other_f1)
        majority_vote_macro_f1.append(f1_score(true_labels, majority_vote_preds, average='macro'))
    model_scores['majority_vote_agg_f1'] = sum(majority_vote_agg_f1) / len(majority_vote_agg_f1)
    model_scores['majority_vote_loss_f1'] = sum(majority_vote_loss_f1) / len(majority_vote_loss_f1)
    model_scores['majority_vote_other_f1'] = sum(majority_vote_other_f1) / len(majority_vote_other_f1)
    model_scores['majority_vote_macro_f1'] = sum(majority_vote_macro_f1) / len(majority_vote_macro_f1)

    return model_scores


def majority_vote(preds):
    vote_predictions = []
    preds = np.array(preds)
    for i in range(len(preds[0])):
        predictions = preds[:, i]
        agg_count = sum(predictions == 0)
        loss_count = sum(predictions == 1)
        other_count = sum(predictions == 2)
        most_count = np.max([agg_count, loss_count, other_count])
        if agg_count == most_count:
            vote_predictions.append(0)
        elif loss_count == most_count:
            vote_predictions.append(1)
        else:
            vote_predictions.append(2)
    return vote_predictions


def evaluate_all_model_result():
    model_id2experiment_dir = model_ids()
    num_models = len(model_id2experiment_dir)
    model_performances = {}
    for model_id in range(1, num_models + 1):
        model_performances[model_id] = evaluate_model_result(model_id2experiment_dir[model_id])
        model_performances[model_id]['model_id'] = model_id

    df = pd.DataFrame.from_records([model_performances[model_id] for model_id in range(1, num_models + 1)])
    df = df.multiply(100) #multiply all f-scores by 100
    df = df.round(1) #round the 100 * f-scores to 1 digit after decimal point
    df.to_csv("../showable/model_performances.csv", index=False)

evaluate_all_model_result()
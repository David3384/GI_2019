import numpy as np
from util import trim
import pandas as pd
from data_loader import Data_loader
from collections import defaultdict
import pickle as pkl
from generator_util import pad_elmo_representation
from load_final_models import load_model
from model_test_data_loader import load_model_data
import os
from model_info import get_model_info
from argparse import ArgumentParser

word2id = pkl.load(open("../model/word.pkl", 'rb'))
id2word = dict([(word2id[key]['id'], key.decode()) for key in word2id.keys()])

class LIME:
    def __init__(self, model_predict, model_threshold, output_dir, input_format, tweet_records, truth_label, pad_elmo=False, unigram_observe_ids=None):
        #model_predict is a function the takes X and evaluates the score, abstracted to keep LIME decoupled from model
        #architecture, input format and use of context features.
        self.dl = Data_loader(labeled_only=True, option='both')
        self.model_predict = model_predict
        self.model_threshold = model_threshold
        self.output_dir = output_dir
        self.input_format = input_format
        self.pad_elmo = pad_elmo
        self.unigram_observe_ids = unigram_observe_ids

        self.tweet_records, self.truth_label = tweet_records, truth_label
        self.scores = self.model_predict(self.tweet_records).flatten()
        self.label_prediction = [1 if self.scores[idx] >= self.model_threshold else 0 for idx in range(len(self.scores))]
        idx_considered = [idx for idx in range(len(self.label_prediction)) if self.label_prediction[idx] == 1]
        self.tweet_id_considered = [self.tweet_records['tweet_id'][idx] for idx in idx_considered]
        included_tweet_records = {}

        for key in self.tweet_records.keys():
            if key == 'word_content_input_elmo' and pad_elmo is False:
                included_tweet_records[key] = [self.tweet_records[key][idx] for idx in idx_considered]
            else:
                included_tweet_records[key] = np.array([self.tweet_records[key][idx] for idx in idx_considered])

        self.tweet_records = included_tweet_records
        self.scores = np.array([self.scores[idx] for idx in idx_considered])

    # tweet_dict is a map from keys to numpy arrays
    # one of the keys is "tweet_id" s.t. it can be mapped back to the original tweet id
    def create_perturbation_samples(self, tweet_dict, elmo_masked_idx=None):
        # perturbed_tests is a
        perturbed_tests = dict([(key, []) for key in tweet_dict])
        p_test_idx = 0

        # (tweet_id, word_idx, 'uni'/'bi') mapped to index in the test batch
        tweet_idx_word_idx2idx, idx2sent_length = {}, {}

        for idx in range(len(tweet_dict['tweet_id'])):
            content_input = tweet_dict['word_content_input'][idx]
            sentence_length = sum([1 if w != 0 else 0 for w in content_input])
            idx2sent_length[idx] = sentence_length

            # mask each unigram
            for word_idx in range(sentence_length):
                tweet_idx_word_idx2idx[(idx, word_idx, 'uni')] = p_test_idx
                p_test_idx += 1

                # prepare corresponding input for each key
                for key in perturbed_tests:
                    if key != 'word_content_input' and key != 'word_content_input_elmo':
                        perturbed_tests[key].append(tweet_dict[key][idx])
                    elif key == 'word_content_input':
                        perturbed_content_input = np.array(tweet_dict[key][idx])
                        perturbed_content_input[word_idx] = 1
                        perturbed_tests[key].append(perturbed_content_input)
                    else: #key = 'word_content_input_elmo'
                        if elmo_masked_idx is None:
                            masked_idx = (word_idx)
                        else:
                            if word_idx == elmo_masked_idx[idx]:
                                masked_idx = (word_idx)
                            else:
                                masked_idx = tuple(sorted((word_idx, elmo_masked_idx[idx])))
                        tweet_id = tweet_dict['tweet_id'][idx]
                        data = pkl.load(open("../data/DS_ELMo_rep_all/%d.pkl" % tweet_id, 'rb'))
                        elmo_masked = data[masked_idx]
                        if self.pad_elmo: # if cnn needs to pad to max_len to keep shape of all inputs the same
                            elmo_masked = pad_elmo_representation(elmo_masked)
                        perturbed_tests[key].append(elmo_masked)
        
        for key in perturbed_tests:
            if key != 'word_content_input_elmo' or self.pad_elmo is True:
                perturbed_tests[key] = np.array(perturbed_tests[key])

        return tweet_idx_word_idx2idx, perturbed_tests, idx2sent_length


    def analyze_perturbations_influence(self, tweet_idx_word_idx2idx, perturbed_tests, idx2sent_length, round,
                                        observe_word_position_idx=None, observe_word_ids=None):
        """
        For first round, if observe_word_id is not None, will keep track of the rank of the unigram specified by
        observe_word_id (for example 9 corresponds with "a") in the sorted order of LIME influence from most influential
        to least influential. For second round, if observe_word_position_idx is not None, then keep track of the rank
        of the word in the tweet at position specified by observe_word_position_idx in the sorted order of LIME influence
        for consistency check.
        """
        if round == 1:
            self.scores = self.model_predict(self.tweet_records).flatten()
            first_round_unigram_ranking = {}
            for observe_word_id in observe_word_ids:
                first_round_unigram_ranking[observe_word_id] = []

        elif round == 2:
            self.scores = self.model_predict(self.masked_tweet_records).flatten()
            second_round_ranking = []

        preturbed_preds = self.model_predict(perturbed_tests).flatten()
        idx2max_min_wordidx = {}
        max_influences = []
        all_influences = []

        for idx in range(len(idx2sent_length)):
            #unigram influence analysis
            influences = []
            for word_idx in range(idx2sent_length[idx]):
                p_test_idx = tweet_idx_word_idx2idx[(idx, word_idx, 'uni')]
                influence = self.scores[idx] - preturbed_preds[p_test_idx]
                influences.append(influence)
            influences = np.array(influences)
            all_influences.append(influences)

            if round == 1 and observe_word_ids is not None:
                tweet_int_arr = self.tweet_records['word_content_input'][idx]
                arg_sort = np.argsort(influences)[::-1]

                for observe_word_id in observe_word_ids:
                    unigram_in_tweet = False
                    for i in range(len(arg_sort)):
                        if tweet_int_arr[arg_sort[i]] == observe_word_id:
                            first_round_unigram_ranking[observe_word_id].append(i)
                            unigram_in_tweet = True
                            break
                    if unigram_in_tweet is False:
                        first_round_unigram_ranking[observe_word_id].append(-1)

            if round == 2:
                arg_sort = np.argsort(influences)[::-1]
                assert observe_word_position_idx[idx] in arg_sort
                for rank_idx in range(idx2sent_length[idx]):
                    if arg_sort[rank_idx] == observe_word_position_idx[idx]:
                        second_round_ranking.append(rank_idx)

            max_influence_word_idx = np.argmax(influences)
            min_influence_word_idx = np.argmin(np.abs(influences))
            max_influences.append(max(influences))
            idx2max_min_wordidx[idx] = (idx2sent_length[idx], max_influence_word_idx, min_influence_word_idx)

        if round == 1:
            return idx2max_min_wordidx, first_round_unigram_ranking, max_influences, all_influences
        elif round == 2:
            return idx2max_min_wordidx, second_round_ranking, max_influences, all_influences

    def lime(self):
        tweet_idx_word_idx2idx, perturbed_tests, idx2sent_length = self.create_perturbation_samples(self.tweet_records)

        (idx2max_min_wordidx, first_round_unigram_ranking,
         first_round_max_influences, first_round_all_influences) \
            = self.analyze_perturbations_influence(tweet_idx_word_idx2idx, perturbed_tests,
                                                   idx2sent_length, round=1,
                                                   observe_word_ids=self.unigram_observe_ids)

        self.masked_tweet_records = {}
        for key in self.tweet_records.keys():
            if key != 'word_content_input_elmo' or self.pad_elmo is True:
                self.masked_tweet_records[key] = np.array(self.tweet_records[key])
            else:
                self.masked_tweet_records[key] = self.tweet_records[key]

        for idx in range(len(idx2max_min_wordidx)):
            self.masked_tweet_records['word_content_input'][idx][idx2max_min_wordidx[idx][2]] = 1 #mask insignificant unigram

        if self.input_format == 'discrete':
            tweet_idx_word_idx2idx, perturbed_tests, idx2sent_length = self.create_perturbation_samples(self.masked_tweet_records)
        else:
            elmo_masked_wordidx = [idx2max_min_wordidx[idx][2] for idx in range(len(idx2max_min_wordidx))]
            tweet_idx_word_idx2idx, perturbed_tests, idx2sent_length = self.create_perturbation_samples(self.masked_tweet_records, elmo_masked_idx=elmo_masked_wordidx)

        observe_word_idx = {}
        for idx in range(len(idx2max_min_wordidx)):
            observe_word_idx[idx] = idx2max_min_wordidx[idx][1]
        second_round_idx2max_min_wordidx, second_round_ranking, second_round_max_influences, second_round_all_influences = \
            self.analyze_perturbations_influence(tweet_idx_word_idx2idx, perturbed_tests, idx2sent_length,
                                                 round=2, observe_word_position_idx=observe_word_idx)

        data = {}
        data['original tweet'] = [self.dl.convert2unicode(trim(arr)) for arr in self.tweet_records['word_content_input']]
        data['masked tweet'] = [self.dl.convert2unicode(trim(arr)) for arr in self.masked_tweet_records['word_content_input']]
        data['first round influences'] = first_round_all_influences
        data['first round max influential unigram'] = [self.dl.convert2unicode([self.tweet_records['word_content_input'][idx][idx2max_min_wordidx[idx][1]]]) for idx in range(len(idx2sent_length))]
        data['first round most insignificant unigram'] = [self.dl.convert2unicode([self.tweet_records['word_content_input'][idx][idx2max_min_wordidx[idx][2]]]) for idx in range(len(idx2sent_length))]
        data['first round max influence'] = first_round_max_influences
        data['second round influences'] = second_round_all_influences
        data['second round most influential unigram'] = [self.dl.convert2unicode([self.tweet_records['word_content_input'][idx][second_round_idx2max_min_wordidx[idx][1]]]) for idx in range(len(idx2sent_length))]
        data['second round max influence'] = second_round_max_influences
        data['first round max influential unigram ranking in second round'] = second_round_ranking
        if self.unigram_observe_ids is not None:
            for unigram_id in self.unigram_observe_ids:
                data['first round unigram %s ranking' % id2word[unigram_id]] = first_round_unigram_ranking[unigram_id]
        pd.DataFrame.from_dict(data).to_csv(self.output_dir, index=False)

        second_round_rank_stats = defaultdict(int)
        for num in second_round_ranking:
            second_round_rank_stats[num] += 1

        #first_round_unigram_ranking uses -1 to indicate that the specified unigram not in the tweet
        #filter out these ranking -1 to get rankings for only those tweets that include the specified unigram
        first_round_unigram_ranking_included = {}
        for unigram_id in self.unigram_observe_ids:
            first_round_unigram_ranking_included[unigram_id] = []
            for i in first_round_unigram_ranking[unigram_id]:
                if i != -1:
                    first_round_unigram_ranking_included[unigram_id].append(i)


        first_round_rank_stats = {}
        for unigram_id in self.unigram_observe_ids:
            stats = defaultdict(int)
            for num in first_round_unigram_ranking_included[unigram_id]:
                stats[num] += 1
            first_round_rank_stats[unigram_id] = stats

        return {
            'unigram_rank_stats': first_round_rank_stats,
            'lime_consistency_stats': second_round_rank_stats,
            'first_round_all_influences': first_round_all_influences,
            'correspondence': self.tweet_id_considered
        }


def lime_analysis(unigram_observe_ids, num_folds, num_runs, num_models):
    model_info = get_model_info(num_folds, num_runs, num_models)
    analysis_keys = ['cv_test', 'heldout', 'ensemble']
    thresholds = model_info['thresholds']
    lime_results = {}
    for run_idx in range(num_runs):
        for fold_idx in range(num_folds):
            for model_id in range(1, num_models + 1):
                if model_id == 5:
                    continue
                for class_idx in range(2):
                    pkl_dir = '../results/model%drun%dfold%dclass%d.pkl' % (model_id, run_idx, fold_idx, class_idx)
                    if os.path.exists(pkl_dir):
                        print('Result is already saved in %s, continue on next.' % pkl_dir)
                        continue
                    print('LIME for %s.' % pkl_dir)
                    load_model_data(model_id)
                    model = load_model(model_id, run_idx, fold_idx, class_idx)

                    input_format = 'discrete' if model_id in model_info['discrete_input'] else 'both'
                    pad_elmo = (model_id in model_info['pad_elmo'])

                    results = {}
                    for analysis_key in analysis_keys:
                        # LIME analysis on cross validation test set
                        prefix = analysis_key
                        if analysis_key == 'cv_test':
                            prefix += '_' + str(fold_idx)
                        tweet_records = pkl.load(open("../data/%s_tweet_X.pkl" % prefix, 'rb'))
                        truth_label = pkl.load(open("../data/%s_truth_y.pkl" % prefix, 'rb'))
                        lime = LIME(
                            model.predict, model_threshold=thresholds[(model_id, run_idx)][class_idx][fold_idx],
                            output_dir = "../showable/model%d_run%d_fold%d_class%d_%s.csv"
                                         % (model_id, run_idx, fold_idx, class_idx, analysis_key),
                            input_format=input_format,
                            tweet_records=tweet_records,
                            truth_label=truth_label,
                            pad_elmo=pad_elmo,
                            unigram_observe_ids=unigram_observe_ids
                        )
                        results[analysis_key] = lime.lime()

                    pkl.dump(results, open(pkl_dir,'wb'))
                    lime_results[(model_id, run_idx, fold_idx, class_idx)] = results

    pkl.dump(lime_results, open("../data/lime_results.pkl", 'wb'))

def get_lime_arguments():
    # parsing the command line argument
    parser = ArgumentParser()
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--num_models', type=int, default=12)
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    args = get_lime_arguments()
    data_loader = Data_loader(labeled_only=True, option='word')
    stopwords = ['a', 'on', 'da', 'into', 'of', 'that']
    tokens_analyzed = [data_loader.convert2int_arr(stopword)[0] for stopword in stopwords]
    lime_analysis(
        tokens_analyzed,
        num_folds=args['num_folds'],
        num_runs=args['num_runs'],
        num_models=args['num_models']
    )
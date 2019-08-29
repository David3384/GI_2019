from ELMo import create_bilm_from_args
import pickle as pkl
from data_loader import Data_loader
import pandas as pd
from util import trim
import numpy as np
from model_test_data_loader import load_model_tweet_dicts, load_model_data
from copy import deepcopy
from load_final_models import load_model
from model_info import get_model_info
from logistic_regression import Logistic_regr
from create_ELMo_embedding import create_adversarial_ELMo_representation


class Adversarial_generator():

    def __init__(self, dataset='labeled'):
        bilm_args = pkl.load(open('../experiments/ELMo_weights/4-23-9pm.param', 'rb'))
        bilm_args['experiment_path'] = 'ELMo_weights/4-23-9pm'
        self.bilm = create_bilm_from_args(bilm_args)
        self.dataset = dataset
        if dataset == 'labeled':
            self.dl = Data_loader(labeled_only=True, option='both')
        else:
            self.dl = Data_loader(labeled_only=False, option='both')


    def compute_log_prob(self, sentences_int_arr):
        tokens = self.bilm.dg.transform_sentences(sentences_int_arr)
        loss = self.bilm.compute_loss_on_data(tokens)
        return -loss


    def sanity_check(self):
        # For each two adjacent tweets, switch the word on every positions and see if both tweets' log probability
        # decrease most of the time
        tweet_ids = list(self.dl.data['data'].keys())
        count_prob_decrease = 0 # number of times the revised sentence has lower probability than original sentence
        count_prob_increase = 0 # number of times the revised sentence has higher probability than original sentence
        prob_increase_samples = {}
        prob_increase_samples['original'] = []
        prob_increase_samples['revised'] = []
        prob_increase_samples['original score'] = []
        prob_increase_samples['revised score'] = []

        for idx in range(len(tweet_ids) - 1):
            tweet_id1 = tweet_ids[idx]
            tweet_id2 = tweet_ids[idx + 1]

            sentence1 = trim(self.dl.data['data'][tweet_id1]['word_padded_int_arr'])
            sentence2 = trim(self.dl.data['data'][tweet_id2]['word_padded_int_arr'])

            log_prob_sentence1 = self.compute_log_prob([sentence1])
            log_prob_sentence2 = self.compute_log_prob([sentence2])
            for word_idx in range(min(len(sentence1), len(sentence2))):
                # swap the two sentences word on this position
                sentence1[word_idx], sentence2[word_idx] = sentence2[word_idx], sentence1[word_idx]
                log_prob_revised_sentence1 = self.compute_log_prob([sentence1])
                log_prob_revised_sentence2 = self.compute_log_prob([sentence2])
                if log_prob_revised_sentence1 <= log_prob_sentence1:
                    count_prob_decrease += 1
                else:
                    count_prob_increase += 1
                    prob_increase_samples['revised'].append(self.dl.convert2unicode(sentence1))
                    prob_increase_samples['revised score'].append(log_prob_revised_sentence1)
                    prob_increase_samples['original score'].append(log_prob_sentence1)

                if log_prob_revised_sentence2 <= log_prob_sentence2:
                    count_prob_decrease += 1
                else:
                    count_prob_increase += 1
                    prob_increase_samples['revised'].append(self.dl.convert2unicode(sentence2))
                    prob_increase_samples['revised score'].append(log_prob_revised_sentence2)
                    prob_increase_samples['original score'].append(log_prob_sentence2)

                # recover the original sentence
                sentence1[word_idx], sentence2[word_idx] = sentence2[word_idx], sentence1[word_idx]
                if log_prob_revised_sentence1 > log_prob_sentence1:
                    prob_increase_samples['original'].append(self.dl.convert2unicode(sentence1))
                if log_prob_revised_sentence2 > log_prob_sentence2:
                    prob_increase_samples['original'].append(self.dl.convert2unicode(sentence2))

            if idx % 10 == 0:
                print("increase: ", count_prob_decrease)
                print("decrease: ", count_prob_increase)
            if idx > 100:
                break
        print("Probability decrease: ", count_prob_decrease)
        print("Probability increase: ", count_prob_increase)
        pd.DataFrame.from_dict(prob_increase_samples).to_csv("../showable/ELMo_sanity_check.csv", index=False)


    def create_natural_sentences(self, mode, token, tweet_dicts):
        assert mode in ['insert', 'replace']
        token_id = self.dl.token2property[token.encode("utf-8")]['id']
        sentence_outputs = {}
        keys = ['original_sentence', 'generated_sentence', 'original_prob', 'generated_prob',
                'original_int_arr', 'generated_int_arr', 'tweet_id']
        for key in keys:
            sentence_outputs[key] = []

        for tweet_id in tweet_dicts.keys():
            sentence = tweet_dicts[tweet_id]['word_padded_int_arr']
            num_words = sum([x != 0 for x in sentence])

            if mode == 'insert':
                if num_words == 50: #already max length, cannot add more words
                    continue
                idx_range = range(num_words + 1)
            else:
                idx_range = range(num_words)

            sentence_outputs['original_int_arr'].append(np.array(sentence))
            original_sentence_unicode = self.dl.convert2unicode(trim(sentence))
            sentence_outputs['original_sentence'].append(original_sentence_unicode)
            original_sentence_prob = self.compute_log_prob([trim(sentence)])
            sentence_outputs['original_prob'].append(original_sentence_prob)
            sentence_outputs['tweet_id'].append(tweet_id)

            max_generated_prob = - np.inf
            most_natural_generated_sentence = None

            for pos in idx_range:
                if mode == 'insert':
                    generated_sentence = insert_element(sentence, pos, token_id)
                else:
                    generated_sentence = np.array(sentence)
                    generated_sentence[pos] = token_id

                new_sentence_prob = self.compute_log_prob([trim(generated_sentence)])
                if new_sentence_prob > max_generated_prob:
                    max_generated_prob = new_sentence_prob
                    most_natural_generated_sentence = generated_sentence

            most_natural_revised_sentence_unicode = self.dl.convert2unicode(trim(most_natural_generated_sentence))
            sentence_outputs['generated_sentence'].append(most_natural_revised_sentence_unicode)
            sentence_outputs['generated_prob'].append(max_generated_prob)
            sentence_outputs['generated_int_arr'].append(np.array(most_natural_generated_sentence))

            if len(sentence_outputs['generated_int_arr']) % 100 == 0:
                print(len(sentence_outputs['generated_int_arr']))
                pkl.dump(sentence_outputs, open("../adversarial_data/%s_%s_natural_sentence_%s.pkl" % (mode, token, self.dataset), 'wb'))

        #order the records in order of maximum probability increase to minimum probability increase
        prob_diff = np.array(sentence_outputs['generated_prob']) - np.array(sentence_outputs['original_prob'])
        sorted_idx = np.argsort(prob_diff)[::-1]
        for key in sentence_outputs.keys():
            sentence_outputs[key] = [sentence_outputs[key][idx] for idx in sorted_idx]
        sentence_outputs['prob_change'] = np.array(sentence_outputs['generated_prob']) - np.array(sentence_outputs['original_prob'])
        pd.DataFrame.from_dict(sentence_outputs).to_csv("../showable/%s_%s_natural_sentence_%s.csv" % (mode, token, self.dataset), index=False)
        pkl.dump(sentence_outputs, open("../adversarial_data/%s_%s_natural_sentence_%s.pkl" % (mode, token, self.dataset), 'wb'))


    def generate_natural_tweets(self, mode, token):
        tweet_dicts = self.dl.data['data']
        self.create_natural_sentences(mode, token, tweet_dicts)


    def evaluate_logistic_regression_prediction(self, mode):
        assert mode in ['score', 'binary']

        lr = Logistic_regr(mode='eval')
        generated_sentences = pkl.load(open("../data/insert_a_natural_sentence.pkl", 'rb'))
        original_int_arrs = generated_sentences['original_int_arr']
        generated_int_arrs = generated_sentences['generated_int_arr']

        if mode == 'score':
            original_agg_scores, original_loss_scores = lr.predict(original_int_arrs, mode="score")
            generated_agg_scores, generated_loss_scores = lr.predict(generated_int_arrs, mode="score")
            return original_agg_scores, original_loss_scores, generated_agg_scores, generated_loss_scores
        else:
            original_agg_labels, original_loss_labels = lr.predict(original_int_arrs, mode="binary")
            generated_agg_labels, generated_loss_labels = lr.predict(generated_int_arrs, mode="binary")
            new_agg_positive_tweet_ids = []
            for idx in range(len(original_agg_labels)):
                if original_agg_labels[idx] == 0 and generated_agg_labels[idx] == 1:
                    new_agg_positive_tweet_ids.append(generated_sentences['tweet_id'][idx])
            new_loss_positive_tweet_ids = []
            for idx in range(len(original_loss_labels)):
                if original_loss_labels[idx] == 0 and generated_loss_labels[idx] == 1:
                    new_loss_positive_tweet_ids.append(generated_sentences['tweet_id'][idx])
            return new_agg_positive_tweet_ids, new_loss_positive_tweet_ids


    def evaluate_model_prediction(self, token, model_id, run_idx, fold_idx, class_idx, mode='binary', top_num=800):
        generated_sentences = pkl.load(open("../adversarial_data/insert_%s_natural_sentence_labeled.pkl" % token, 'rb'))
        original_int_arrs = generated_sentences['original_int_arr'][:top_num]
        revised_int_arrs = generated_sentences['generated_int_arr'][:top_num]
        tweet_ids = generated_sentences['tweet_id'][:top_num]

        all_tweets = self.dl.all_data()
        original_tweets = []
        generated_tweets = []

        tweetid2tweetidx = {}
        for idx in range(len(all_tweets)):
            tweetid2tweetidx[all_tweets[idx]['tweet_id']] = idx

        for idx in range(len(original_int_arrs)):
            tweet = all_tweets[tweetid2tweetidx[tweet_ids[idx]]]
            original_tweets.append(tweet)
            generated_tweet = deepcopy(tweet)
            assert np.all(generated_tweet['word_padded_int_arr'] == original_int_arrs[idx])
            generated_tweet['word_padded_int_arr'] = revised_int_arrs[idx]
            generated_tweet['word_int_arr'] = trim(generated_tweet['word_padded_int_arr'])
            generated_tweets.append(generated_tweet)

        generated_elmo_dir = None
        original_elmo_dir = None
        if model_id in (3, 4, 6, 7): #DS ELMo
            generated_elmo_dir = "../adversarial_data/DS_ELMo_adversarial_insert_%s" % token
            original_elmo_dir = "../data/DS_ELMo_rep"
        if model_id == 5: #NonDS ELMo
            generated_elmo_dir = "../adversarial_data/NonDS_ELMo_adversarial_insert_%s" % token
            original_elmo_dir = "../data/NonDS_ELMo_rep"

        load_model_tweet_dicts(model_id, generated_tweets, elmo_dir=generated_elmo_dir)
        generated_tweet_X = pkl.load(open("../data/adversarial_tweet_X.pkl", 'rb'))

        load_model_tweet_dicts(model_id, original_tweets, elmo_dir=original_elmo_dir)
        original_tweet_X = pkl.load(open("../data/adversarial_tweet_X.pkl", 'rb'))

        model = load_model(model_id, run_idx, fold_idx, class_idx)
        original_predictions = model.predict(original_tweet_X)
        generated_predictions = model.predict(generated_tweet_X)

        assert mode in ['score', 'binary']
        if mode == 'score': # analyze prediction numerical score change
            return original_predictions, generated_predictions

        else:  # analyze label flipping
            threshold = get_model_info(num_runs=5, num_folds=5, num_models=model_id)['thresholds'][(model_id, run_idx)][class_idx][fold_idx]
            original_pred_labels = [1 if x >= threshold else 0 for x in original_predictions]
            generated_pred_labels = [1 if x >= threshold else 0 for x in generated_predictions]
            new_positive_tweet_ids = []
            new_negative_tweet_ids = []

            for idx in range(len(original_predictions)):
                if original_pred_labels[idx] == 0 and generated_pred_labels[idx] == 1:
                    new_positive_tweet_ids.append(original_tweets[idx]['tweet_id'])
                if original_pred_labels[idx] == 1 and generated_pred_labels[idx] == 0:
                    new_negative_tweet_ids.append(original_tweets[idx]['tweet_id'])
            return len(new_positive_tweet_ids)


    def evaluate_all_models(self, token, class_idx):
        results = {}
        for model_id in [1, 2, 18, 19]:
            flipped_counts = []
            for fold_idx in range(5):
                counts = []
                for run_idx in range(5):
                    counts.append(self.evaluate_model_prediction(token, model_id, run_idx, fold_idx, class_idx))
                flipped_counts.append(sum(counts) / len(counts))
            results[model_id] = sum(flipped_counts) / len(flipped_counts)
        pkl.dump(results, open("../adversarial_data/insert_%s_model_stats_labeled_121819.pkl" % token, 'wb'))
        analysis_dict = {}
        analysis_dict['model_id'] = sorted([x for x in results.keys()])
        analysis_dict['num_flipped_adversarials'] = [results[x] for x in analysis_dict['model_id']]
        pd.DataFrame.from_dict(analysis_dict).to_csv("../showable/adversarial_%s_stats_labeled.csv" % token, index=False)


#====================================
# helper functions

def insert_element(list, position, inserted_element):
    # Insert an element in the position of the original list, return a new list
    new_list = np.array([0] * 50)
    for idx in range(0, position):
        new_list[idx] = list[idx]
    new_list[position] = inserted_element
    for idx in range(position + 1, 50):
        new_list[idx] = list[idx - 1]
    return new_list


if __name__ == '__main__':
    stopwords = ['a', 'on', 'da', 'into', 'of', 'that']
    adversarial_gen = Adversarial_generator(dataset='labeled')
    for idx in range(0, len(stopwords)):
        adversarial_gen.generate_natural_tweets(mode='insert', token=stopwords[idx])
        adversarial_gen.evaluate_all_models(token=stopwords[idx], class_idx=0)
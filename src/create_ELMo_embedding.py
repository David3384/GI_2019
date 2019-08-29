from ELMo import create_bilm_from_args
import pickle as pkl
import os
import numpy as np
from embedwELMo import embed_one_sentence
from util import trim


def create_adversarial_ELMo_representation(domain_specific, input_file, output_dir, parameter_dir=None):
    generated_sentences = pkl.load(open(input_file, 'rb'))
    revised_int_arrs = generated_sentences['generated_int_arr']
    tweet_ids = generated_sentences['tweet_id']
    tweet_dict = {}
    for idx in range(len(tweet_ids)):
        tweet_dict[tweet_ids[idx]] = {}
        tweet_dict[tweet_ids[idx]]['word_int_arr'] = trim(revised_int_arrs[idx])
    create_ELMo_representation(tweet_dict, domain_specific=domain_specific, output_dir=output_dir, parameter_dir=parameter_dir)


def create_all_tweets_ELMo_representation(domain_specific, output_dir, masked_unigram_id=None, parameter_dir=None):
    data = pkl.load(open('../data/labeled_data.pkl', 'rb'))
    print('Loading data finished.')
    tweet_dicts = data['data']
    create_ELMo_representation(tweet_dicts, domain_specific, output_dir, masked_unigram_id, parameter_dir)


def create_ELMo_representation(tweet_dicts, domain_specific, output_dir, masked_unigram_id=None, parameter_dir=None):
    """
    Create ELMo representation for all labeled tweets with a certain unigram masked as UNK. The ELMo representations will
    be stored as .npy files for each tweet.
    """
    if domain_specific:
        args = {}
        args['experiment_path'] = parameter_dir
        path = args['experiment_path']
        args = pkl.load(open('../experiments/%s.param' % path, 'rb'))
        args['experiment_path'] = path
        bilm = create_bilm_from_args(args)
    else:
        words = pkl.load(open('../model/word.pkl', 'rb'))
        id2word = dict([(words[w]['id'], w.decode()) for w in words])

    data_dir = '../data/' + output_dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    for tweet_id in tweet_dicts:
        word_int_arr = trim(tweet_dicts[tweet_id]['word_int_arr'][:50])

        #mask the specified unigram as UNK if masked_unigram_id is specified
        if masked_unigram_id is not None:
            for word_idx in range(len(word_int_arr)):
                if word_int_arr[word_idx] == masked_unigram_id:
                    word_int_arr[word_idx] = 1

        #use DS ELMo/NonDS ELMo code to generate corresponding elmo representation
        if domain_specific:
            elmo_rep = np.array([x.detach().cpu().numpy()[0] for x in bilm.create_rep([word_int_arr])])
        else:
            sentence = [id2word[idx] for idx in word_int_arr]
            elmo_rep = embed_one_sentence(sentence)
        np.save("%s%d.npy" % (data_dir, tweet_id), elmo_rep)


def create_LIME_ELMo_representation(domain_specific, output_dir, parameter_dir=None):
    """
    Create ELMo representation for all tweets, including ELMo representation of original tweet, tweet masked for each
    unigram and tweet masked for any pair of two unigrams. This complete ELMo representation is used for LIME analysis.
    """
    data = pkl.load(open('../data/labeled_data.pkl', 'rb'))
    print('Loading data finished.')
    tweet_dicts = data['data']

    if domain_specific is False:
        words = pkl.load(open('../model/word.pkl', 'rb'))
        id2word = dict([(words[w]['id'], w.decode()) for w in words])
    else:
        assert parameter_dir is not None
        args = {}
        args['experiment_path'] = parameter_dir
        path = args['experiment_path']
        args = pkl.load(open('../experiments/%s.param' % path, 'rb'))
        args['experiment_path'] = path

        bilm = create_bilm_from_args(args)

    data_dir = '../data/' + output_dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    for tweet_id in tweet_dicts:
        tweet_elmo_property = {}
        word_int_arr = tweet_dicts[tweet_id]['word_int_arr'][:50]
        if domain_specific:
            tweet_elmo_property['original'] = np.array([x.detach().cpu().numpy()[0] for x in bilm.create_rep([word_int_arr])])
        else:
            sentence = [id2word[idx] for idx in word_int_arr]
            tweet_elmo_property['original'] = embed_one_sentence(sentence)

        #mask one unigram
        for mask_word_idx in range(len(word_int_arr)):
            masked_int_arr = np.array(word_int_arr)
            masked_int_arr[mask_word_idx] = 1 #mask as unknown
            if domain_specific:
                tweet_elmo_property[mask_word_idx] = np.array([x.detach().cpu().numpy()[0] for x in bilm.create_rep([masked_int_arr])])
            else:
                sentence = [id2word[idx] for idx in masked_int_arr]
                tweet_elmo_property[mask_word_idx] = embed_one_sentence(sentence)

        #mask two unigrams
        for mask_word_idx1 in range(len(word_int_arr) - 1):
            for mask_word_idx2 in range(mask_word_idx1 + 1, len(word_int_arr)):
                masked_int_arr = np.array(word_int_arr)
                masked_int_arr[mask_word_idx1] = 1
                masked_int_arr[mask_word_idx2] = 1
                if domain_specific:
                    tweet_elmo_property[(mask_word_idx1, mask_word_idx2)] = np.array([x.detach().cpu().numpy()[0] for x in bilm.create_rep([masked_int_arr])])
                else:
                    sentence = [id2word[idx] for idx in masked_int_arr]
                    tweet_elmo_property[(mask_word_idx1, mask_word_idx2)] = embed_one_sentence(sentence)

        pkl.dump(tweet_elmo_property, open("%s%d.pkl" % (data_dir, tweet_id), 'wb'))



if __name__ == '__main__':
    #create_adversarial_ELMo_representation(domain_specific=True, input_file="../data/insert_on_natural_sentence.pkl",
    #                                        output_dir="DS_ELMo_adversarial_insert_on/", parameter_dir="ELMo_weights/4-23-9pm")
    #create_ELMo_representation(domain_specific=False, output_dir="Non_DS_ELMo_rep/")
    #create_masked_unigram_ELMo_representation("DS_ELMo_rep_masked_a/", parameter_dir="ELMo_weights/4-23-9pm",
    #                                          masked_unigram_id=9)
    from data_loader import Data_loader
    dl_word = Data_loader(labeled_only=True, option='word')

    stopwords = ['o', '...lol', 'let', 'yeah', 'got', 'any', 'into', 'thats', 'who', 'out', 'that', "'s",
                 'yo', 'as', 'we', 'be', 'of', 'u', 'do', 'in']
    for stopword in stopwords:
        unigram_id = dl_word.convert2int_arr(stopword)[0]
        if unigram_id != 1: #the stopword is in the vocabulary
            create_all_tweets_ELMo_representation(domain_specific=True, output_dir="DS_ELMo_rep_masked_%s/" % stopword, parameter_dir="ELMo_weights/4-23-9pm",
                                              masked_unigram_id=unigram_id)
    #create_masked_unigram_ELMo_representation("DS_ELMo_rep_masked_da/", parameter_dir="ELMo_weights/4-23-9pm",
    #                                          masked_unigram_id=24)
    #create_masked_unigram_ELMo_representation(domain_specific=False, output_dir="NonDS_ELMo_rep_masked_a/", masked_unigram_id=9)
    #create_masked_unigram_ELMo_representation(domain_specific=False, output_dir="NonDS_ELMo_rep_masked_on/", masked_unigram_id=13)
    #create_masked_unigram_ELMo_representation(domain_specific=False, output_dir="NonDS_ELMo_rep_masked_da/", masked_unigram_id=24)

from bi_lm import BiLM
from argparse import ArgumentParser
import random
import pickle as pkl
import os
import numpy as np

# parsing the command line argument
def get_arguments():
    print('Parsing arguments ...')
    parser = ArgumentParser()
    parser.add_argument('-ml', '--max_len', type=int, default=50)
    parser.add_argument('-v', '--vocab_size', type=int, default=30000)
    parser.add_argument('-wd', '--wembed_dim', type=int, default=300)
    parser.add_argument('-hd', '--hidden_dim', type=int, default=64)
    parser.add_argument('-d', '--depth', type=int, default=2)
    parser.add_argument('-p', '--experiment_path', type=str)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-o', '--option', type=str)
    
    args = vars(parser.parse_args())
    return args

# create language model from arguments
def create_bilm_from_args(args):
    print('Creating model ... ', end='')
    experiment_path = '../experiments/' + args['experiment_path']
    
    bilm = BiLM(experiment_path, args['max_len'], args['vocab_size'], args['wembed_dim'],
                args['hidden_dim'], args['depth'])
    print('finished')
    return bilm

# create unlabeled dataset
def create_unlabeled_dataset():
    print('Creating Language Model training set ... ', end='')
    data = pkl.load(open('../data/data.pkl', 'rb'))
    
    # get train data
    train_sentences = []
    for tweet_id in data['unlabeled_tr']:
        train_sentences.append(data['data'][tweet_id]['word_int_arr'])

    # get dev data
    dev_sentences = []
    for tweet_id in data['unlabeled_val']:
        dev_sentences.append(data['data'][tweet_id]['word_int_arr'])
    print('finished')
    return train_sentences, dev_sentences

# train the langauge model
def train_lm(args):
    bilm = create_bilm_from_args(args)
    train_sentences, dev_sentences = create_unlabeled_dataset()
    bilm.train(train_sentences, dev_sentences, epochs=args['epoch'])

def create_rep(args):
    # load the model and corresponding paths
    path = args['experiment_path']
    args = pkl.load(open('../experiments/%s.param' % path, 'rb'))
    args['experiment_path'] = path
    
    # creating the model
    bilm = create_bilm_from_args(args)
    data_dir = '../data/' + args['experiment_path'] + '/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    data = pkl.load(open('../data/data.pkl', 'rb'))
    print('Loading data finished.')
    for tweet_id in data['data']:
        if data['data'][tweet_id].get('label') is not None:
            rep = np.array([x.detach().cpu().numpy()[0] for x in
                            bilm.create_rep([data['data'][tweet_id]['word_int_arr']])])
            np.save('%s%d' % (data_dir, tweet_id), rep)

def evaluate_prob(args):
    # load the model and corresponding paths
    path = args['experiment_path']
    args = pkl.load(open('../experiments/%s.param' % path, 'rb'))
    args['experiment_path'] = path

    # creating the model
    bilm = create_bilm_from_args(args)
    data_dir = '../data/' + args['experiment_path'] + '/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    data = pkl.load(open('../data/data.pkl', 'rb'))
    print('Loading data finished.')

    for tweet_id in data['data']:
        if data['data'][tweet_id].get('label') is not None:
            prob = np.array([x.detach().cpu().numpy()[0] for x in
                            bilm.evaluate_prob([data['data'][tweet_id]['word_int_arr']])])


if __name__ == '__main__':
    # get arguments
    args = get_arguments()
    print(args)
    
    # either train the model or write the representation
    if args['option'] == 'train':
        train_lm(args)
    elif args['option'] == 'rep':
        create_rep(args)
    elif args['option'] == 'prob':
        evaluate_prob(args)
    else:
        print('option %s not implemented.' % args['option'])


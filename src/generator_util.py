import numpy as np
import random
from util import pad_with_np_zeros

label2class_idx = {
    'Aggression': 0,
    'Loss': 1,
    'Other': 2
}

def helper_generator(X, y, batch_size, word_dropper):
    counter = 0
    for key in X:
        num_data = len(X[key])
    order = [i for i in range(num_data)]
    while True:
        if counter % num_data == 0:
            random.shuffle(order)
        idxes = [order[idx % num_data] for idx in range(counter, counter + batch_size)]
        counter += batch_size
        yield word_dropper.dropout((dict([(key, X[key][idxes]) for key in X]), y[idxes]))

def extract_y_from_tweet_dicts(tweet_dicts):
    return np.array([label2class_idx[tweet_dict['label']] for tweet_dict in tweet_dicts])

def create_data(input_name2id2np, tweet_dicts,
                input_format, class_idx=None,
                batch_size=None, word_dropper=None,
                pad_elmo=False, elmo_dir=None):
    """
    A map that takes in tweet dictionaries and return data points readable for keras fit/fit_generator

    Parameters
    ----------

    tweet_dicts: a list of tweets dictionary
    return_generators: whether (generator, step_size) is returned or (X, y) is returned

    Returns
    -------
    X: key-worded inputs
    y: one-hot labels
        OR
    generator: a generator that will generate
    step_size: number of times for a generator to complete one epoch

    """
    data, keys = [], None
    assert input_format in ['discrete', 'elmo', 'both']

    # convert each tweet_dict to a dictionary that only contains field that is recognizable and useulf
    # for the model
    for tweet_dict in tweet_dicts:
        result = {}
        one_hot_labels = np.eye(3)
        if tweet_dict['label'] == 'Aggression':
            result['y'] = one_hot_labels[0]
        elif tweet_dict['label'] == 'Loss':
            result['y'] = one_hot_labels[1]
        else:
            result['y'] = one_hot_labels[2]
        if class_idx is not None:
            result['y'] = result['y'][class_idx] > 0.5

        if input_format == 'discrete':
            result['word_content_input'] = tweet_dict['word_padded_int_arr']
        else:
            elmo_rep = np.load("%s/%s.npy" % (elmo_dir, str(tweet_dict['tweet_id'])))
            result['word_content_input_elmo'] = elmo_rep
            if pad_elmo:
                result['word_content_input_elmo'] = pad_elmo_representation(result['word_content_input_elmo'])
            result['word_content_input'] = tweet_dict['word_padded_int_arr']

        result['char_content_input'] = tweet_dict['char_padded_int_arr']
        for input_name in input_name2id2np:
            result[input_name + '_input'] = input_name2id2np[input_name][tweet_dict['tweet_id']]
        if 'rationale' in tweet_dict:
            result['rationale_distr'] = np.array(tweet_dict['rationale']) / sum(tweet_dict['rationale'])
        else:
            result['rationale_distr'] = None
        if 'rationale_exclude_UNK' in tweet_dict:
            result['rationale_exclude_UNK_distr'] = np.array(tweet_dict['rationale_exclude_UNK']) / sum(tweet_dict['rationale_exclude_UNK'])
        else:
            result['rationale_exclude_UNK_distr'] = None
        result['tweet_id'] = tweet_dict['tweet_id']
        if keys is None:
            keys = [key for key in result]
        data.append(result)


    X = dict()
    for key in keys:
        if key == 'word_content_input_elmo' and pad_elmo is False:
            X[key] = [d[key] for d in data] #elmo word representations are of different length (not padded)
        else:
            X[key] = np.array([d[key] for d in data])
    y = np.array([d['y'] for d in data])

    # return the entire datapoints and labels in one single array
    if batch_size is None:
        return X, y

    generator = helper_generator(X, y, batch_size, word_dropper)
    step_size = len(data) // batch_size + 1
    return generator, step_size


# takes in tr, val, test, each of it a list of dictionaries
# besides basic y and word/char level input
# sets the input field by input_name2id2np
# create the cv fold of data for tr, val, test
def create_clf_data(input_name2id2np, tr_test_val_dicts,
                    input_format, class_idx,
                    batch_size=None,
                    word_dropper=None,
                    pad_elmo=False, elmo_dir=None):
    tr, val, test = tr_test_val_dicts
    return (create_data(input_name2id2np, tr,
                        input_format=input_format, class_idx=class_idx,
                        batch_size=batch_size, word_dropper=word_dropper,
                        pad_elmo=pad_elmo, elmo_dir=elmo_dir),
            create_data(input_name2id2np, val,
                        input_format=input_format, class_idx=class_idx,
                        pad_elmo=pad_elmo, elmo_dir=elmo_dir),
            create_data(input_name2id2np, test,
                        input_format=input_format, class_idx=class_idx,
                        pad_elmo=pad_elmo, elmo_dir=elmo_dir))


#pad the elmo representation to max_len so that all elmo representation have same input shape (required for keras CNN)
def pad_elmo_representation(elmo_rep):
    elmo_rep = np.pad(elmo_rep, pad_width=((0, 0), (0, 50 - elmo_rep.shape[1]), (0, 0)), mode='constant')
    return elmo_rep


if __name__ == '__main__':
    from data_loader import Data_loader
    option = 'word'
    max_len = 50
    vocab_size = 40000
    dl = Data_loader(vocab_size=vocab_size, max_len=max_len, option=option)
    fold_idx = 0
    data_fold = dl.cv_data(fold_idx)
    tr, val, test = data_fold
    print(tr[0])
    '''
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_clf_data(simplest_tweet2data,
                                                                           data_fold)
    for key in X_train:
        print(X_train[key])
    '''

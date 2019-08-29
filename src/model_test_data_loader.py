import pickle as pkl
from generator_util import create_data
from data_loader import Data_loader


dl = Data_loader(labeled_only='True', option='both')

def load_model_data(model_id):
    """
    Load the cv test data, heldout test data, ensemble data that corresponds with the model's input format.
    """
    assert model_id in [x for x in range(1, 20)]
    if model_id in (1, 2):     #1. CNN + DS Word Embedding + Context Features
        load_data(with_context_features=True, input_format='discrete')

    elif model_id == 3:        # 3. CNN + DS Word Embedding + Context Features + DS ELMo
        load_data(with_context_features=True, input_format="both", pad_elmo=True, elmo_dir="../data/DS_ELMo_rep")

    elif model_id == 4:        # 4. CNN + DS Word Embedding + DS ELMo
        load_data(with_context_features=False, input_format="both", pad_elmo=True, elmo_dir="../data/DS_ELMo_rep")

    elif model_id == 5:        # 5. CNN + DS Word Embedding + Context Features + Non-DS ELMo
        load_data(with_context_features=True, input_format="both", pad_elmo=True, elmo_dir="../data/NonDS_ELMo_rep")

    elif model_id == 6:        # 6. LSTM Attention + DS Word Embedding + DS ELMo + Context Features
        load_data(with_context_features=True, input_format="both", pad_elmo=False, elmo_dir="../data/DS_ELMo_rep")

    elif model_id == 7:        # 7. LSTM Attention + DS Word Embedding + DS ELMo + Context Features + Rationale
        load_data(with_context_features=True, input_format="both", pad_elmo=False, elmo_dir="../data/DS_ELMo_rep")

    #8. LSTM Attention + DS Word Embedding + CF
    #9. LSTM Attention + DS Word Embedding + CF + Rationale
    #10. CNN + DS Word Embedding + CF + KRdropout
    #11. CNN+DSWEMB+CF+CY
    #12. CNN+DSWEMB+CF+DROP
    #13. LSTM Attention + DS Word Embedding + CF + KRdropout
    #14. LSTM Attention + DS Word Embedding + CF + CY
    #15. LSTM Attention + DS Word Embedding + CF + DROP
    elif model_id in range(8, 16):
        load_data(with_context_features=True, input_format="discrete")

    # 16. LSTM Attention + DS Word Embedding
    # 17. LSTM Attention + DS Word Embedding + Rationale
    elif model_id in [16, 17]:
        load_data(with_context_features=False, input_format="discrete")

    # 18. LSTM Attention + DS Word Embedding + CF at last (Compared to 8, -CF before)
    # 19. LSTM Attention + DS Word Embedding + CF at last + Rationale (Compared to 9, -CF before)
    elif model_id in [18, 19]:
        load_data(with_context_features=True, input_format="discrete")


def load_data(with_context_features, input_format, pad_elmo=False, elmo_dir=None):
    """
    Load test data according to with/without context feature, input_format, pad/don't pad elmo representation, elmo_dir.
    Return three kinds of test data: cross validation fold test data, ensemble data, heldout test data
    """
    if with_context_features:
        input_dim2id2np = pkl.load(open("../data/emnlp2018_best.pkl", 'rb'))
    else:
        input_dim2id2np = {}

    cv_test_dicts = [dl.cv_data(fold_idx)[2] for fold_idx in range(5)]
    heldout_test_dict = dl.test_data()
    ensemble_dict = dl.ensemble_data()

    for idx in range(len(cv_test_dicts)):
        cv_test_dict = cv_test_dicts[idx]
        tweet_X, truth_y = create_data(input_name2id2np=input_dim2id2np, tweet_dicts=cv_test_dict,
                                       input_format=input_format, pad_elmo=pad_elmo, elmo_dir=elmo_dir)
        pkl.dump(tweet_X, open("../data/cv_test_%d_tweet_X.pkl" % idx, 'wb'))
        pkl.dump(truth_y, open("../data/cv_test_%d_truth_y.pkl" % idx, 'wb'))

    heldout_tweet_X, heldout_truth_y = create_data(input_name2id2np=input_dim2id2np, tweet_dicts=heldout_test_dict,
                                                   input_format=input_format, pad_elmo=pad_elmo, elmo_dir=elmo_dir)
    pkl.dump(heldout_tweet_X, open("../data/heldout_tweet_X.pkl", 'wb'))
    pkl.dump(heldout_truth_y, open("../data/heldout_truth_y.pkl", 'wb'))

    ensemble_tweet_X, ensemble_truth_y = create_data(input_name2id2np=input_dim2id2np, tweet_dicts=ensemble_dict,
                                                     input_format=input_format, pad_elmo=pad_elmo, elmo_dir=elmo_dir)
    pkl.dump(ensemble_tweet_X, open("../data/ensemble_tweet_X.pkl", 'wb'))
    pkl.dump(ensemble_truth_y, open("../data/ensemble_truth_y.pkl", 'wb'))


def load_tweet_dict(tweet_dicts, with_context_features, input_format, pad_elmo=False, elmo_dir=None):
    if with_context_features:
        input_dim2id2np = pkl.load(open("../data/emnlp2018_best.pkl", 'rb'))
    else:
        input_dim2id2np = {}
    tweet_X, truth_y = create_data(input_name2id2np=input_dim2id2np, tweet_dicts=tweet_dicts,
                                   input_format=input_format, pad_elmo=pad_elmo, elmo_dir=elmo_dir)
    pkl.dump(tweet_X, open("../data/adversarial_tweet_X.pkl", 'wb'))
    pkl.dump(truth_y, open("../data/adversarial_truth_y.pkl", 'wb'))


def load_model_tweet_dicts(model_id, tweet_dicts, elmo_dir=None):
    assert model_id in [x for x in range(1, 20)]
    if model_id in (1, 2):     #1. CNN + DS Word Embedding + Context Features
        load_tweet_dict(tweet_dicts, with_context_features=True, input_format='discrete')

    elif model_id == 3:        # 3. CNN + DS Word Embedding + Context Features + DS ELMo
        load_tweet_dict(tweet_dicts, with_context_features=True, input_format="both", pad_elmo=True, elmo_dir=elmo_dir)

    elif model_id == 4:        # 4. CNN + DS Word Embedding + DS ELMo
        load_tweet_dict(tweet_dicts, with_context_features=False, input_format="both", pad_elmo=True, elmo_dir=elmo_dir)

    elif model_id == 5:        # 5. CNN + DS Word Embedding + Context Features + Non-DS ELMo
        load_tweet_dict(tweet_dicts, with_context_features=True, input_format="both", pad_elmo=True, elmo_dir=elmo_dir)

    elif model_id == 6:        # 6. LSTM Attention + DS Word Embedding + DS ELMo + Context Features
        load_tweet_dict(tweet_dicts, with_context_features=True, input_format="both", pad_elmo=False, elmo_dir=elmo_dir)

    elif model_id == 7:        # 7. LSTM Attention + DS Word Embedding + DS ELMo + Context Features + Rationale
        load_tweet_dict(tweet_dicts, with_context_features=True, input_format="both", pad_elmo=False, elmo_dir=elmo_dir)

    # 8. LSTM Attention + DS Word Embedding + CF
    # 9. LSTM Attention + DS Word Embedding + CF + Rationale
    # 10. CNN + DS Word Embedding + CF + KRdropout
    # 11. CNN+DSWEMB+CF+CY
    # 12. CNN+DSWEMB+CF+DROP
    # 13. LSTM Attention + DS Word Embedding + CF + KRdropout
    # 14. LSTM Attention + DS Word Embedding + CF + CY
    # 15. LSTM Attention + DS Word Embedding + CF + DROP
    elif model_id in range(8, 16):
        load_tweet_dict(tweet_dicts, with_context_features=True, input_format="discrete")

    # 16. LSTM Attention + DS Word Embedding
    # 17. LSTM Attention + DS Word Embedding + Rationale
    elif model_id in [16, 17]:
        load_tweet_dict(tweet_dicts, with_context_features=False, input_format="discrete")

    # 18. LSTM Attention + DS Word Embedding + CF at last (Compared to 8, -CF before)
    # 19. LSTM Attention + DS Word Embedding + CF at last + Rationale (Compared to 9, -CF before)
    elif model_id in [18, 19]:
        load_tweet_dict(tweet_dicts, with_context_features=True, input_format="discrete")


        
if __name__ == '__main__':
    for model_id in range(1, 8):
        print(model_id)
        load_model_data(model_id)
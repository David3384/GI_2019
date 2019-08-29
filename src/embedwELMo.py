from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

NUM_REPS = 3
elmo = Elmo(options_file, weight_file, NUM_REPS, dropout=0)
print('ELMo model loading finished.')

def embed_one_sentence(sentence):
    sentence_batch = [sentence]
    character_ids = batch_to_ids(sentence_batch)
    rep = np.array([elmo(character_ids)['elmo_representations'][rep_idx][0].detach().numpy()
                    for rep_idx in range(NUM_REPS)])
    return rep


def test():
    sentences = [['I', 'dreamed', 'a', 'dream', '.'], ['wait', '.'], ['hello', 'world', 'hey'],
                 ['hello', 'world', 'hey'] * 2, ['hello', 'world', 'hey'] * 3]
    for sentence in sentences:
        embed_one_sentence(sentence)

if __name__ == '__main__':
    import pickle as pkl
    words = pkl.load(open('word.pkl', 'rb'))
    id2word = dict([(words[w]['id'], w.decode()) for w in words])
    data = pkl.load(open('data.pkl', 'rb'))
    for tweet_id in data['data']:
        tweet_dict = data['data'][tweet_id]
        word_int_arr = tweet_dict['word_int_arr']
        if tweet_dict.get('label') is not None:
            sentence = [id2word[idx] for idx in word_int_arr]
            rep = embed_one_sentence(sentence)
            np.save('orig_ELMo_rep/%d' % tweet_id, rep)
            _, sent_len, _ = rep.shape
            if sent_len != len(word_int_arr):
                print(tweet_id)
                print(sentence)


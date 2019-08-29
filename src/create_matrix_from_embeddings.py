from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pickle as pkl

word = pkl.load(open('../model/word.pkl', 'rb'))
id2word = dict([(word[w]['id'], w) for w in word])


def create_matrix(embed_file_dir, out_dir, vocab_size, embed_dim, id2str=None):
    if id2str is None:
        def id2str(x):
            return str(x)

    # initialize matrix
    result = np.zeros((vocab_size, embed_dim))

    unknown_token_set = set()
    # load embedding
    wv = KeyedVectors.load_word2vec_format(embed_file_dir, binary=True)
    print('loading embeddings done')

    found_words = 0
    avg = np.zeros(embed_dim)
    for _ in range(1, vocab_size):
        try:
            result[_] = wv[id2str(_)]
            avg += result[_]
            found_words += 1
        except:
            unknown_token_set.add(_)
    avg /= found_words
    for _ in unknown_token_set:
        #result[_] = wv[id2str(1)]
        result[_] = avg
    np.savetxt(out_dir, result)
    print(unknown_token_set)
    pkl.dump(unknown_token_set, open(out_dir + '.unknown.pkl', 'wb'))
    print('out of %d words, %d are found' % (vocab_size, found_words))


def tweet_embed2np(embed_file_dir, out_dir, vocab_size, embed_dim):
    result = np.zeros((vocab_size, embed_dim))

    found_words = 0
    unknown_token_set = set()
    w2v = {}
    with open(embed_file_dir, 'r') as in_file:
        for l in in_file:
            tokens = l.split(' ')
            word = tokens[0]
            vec = np.array([float(s) for s in tokens[1:]])
            w2v[word] = vec
    print('tweet embedding load completes')

    avg = np.zeros(embed_dim)

    for _ in range(1, vocab_size):
        try:
            word = id2word[_].decode()
            if word == '@user':
                word = '<user>'
            if word == '!url':
                word = '<url>'
            for n in '0123456789':
                if n in word:
                    word = '<number>'
            if '#' in word:
                word = '<hashtag>'
            result[_] = w2v[word]
            avg += result[_]
            found_words += 1
        except:
            unknown_token_set.add(_)

    avg /= found_words
    for _ in unknown_token_set:
        result[_] = avg
    np.savetxt(out_dir, result)
    pkl.dump(unknown_token_set, open(out_dir + '.unknown.pkl', 'wb'))
    print('out of %d words, %d are found' % (vocab_size, found_words))


if __name__ == '__main__':
    #load tweeter embedding
    #embed_dim = 200
    #vocab_size = 40000
    #embed_file_dir = '../glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt'
    #out_dir = '../weights/word_emb_twitter_' + str(embed_dim) + '.np'
    #m = tweet_embed2np(embed_file_dir, out_dir, vocab_size, embed_dim)

    #load google news embedding
    embed_dim = 300
    vocab_size = 40000
    embed_file_dir = '../GoogleNews-vectors-negative300.bin'
    out_dir = '../weights/word_emb_google_300.np'

    def id2str(x):
        return id2word[x].decode()

    create_matrix(embed_file_dir, out_dir, vocab_size, embed_dim, id2str=id2str)

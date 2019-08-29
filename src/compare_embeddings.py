import csv
from data_loader import Data_loader
from preprocess import preprocess
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pickle

def preprocess_aave():
    path = '../raw_embedding_corpus/TwitterAAE-full-v1/twitteraae_all_aa'
    sentences = []
    with open(path) as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        text_idx = 5
        line = 0
        for row in tsv_reader:
            if line % 10000 == 0: print(line)
            text_bytes = preprocess(row[text_idx])  # unicode, non-readable for w2v
            sentences.append(text_bytes.split(b' '))
            line += 1
    print('Num of sentences:', len(sentences))
    print('Check sentence0:', sentences[0])
    pickle.dump(sentences, open('preprocessed_aave.pkl', 'wb'))

def preprocess_chicago():
    path = '../raw_embedding_corpus/chicago_tweets_900000.csv'
    sentences = []
    with open(path) as f:
        tsv_reader = csv.reader(f, delimiter=',')
        text_idx = 3
        line = 0
        for row in tsv_reader:
            if line == 0: #skip column names
                line += 1
                continue
            if line % 10000 == 0:
                print(line)
            text_bytes = preprocess(row[text_idx])  # unicode, non-readable for w2v
            sentences.append(text_bytes.split(b' '))
            line += 1
    print('Num of sentences:', len(sentences))
    print('Check sentence0:', sentences[0])
    pickle.dump(sentences, open('preprocessed_chicago.pkl', 'wb'))

def unicode_to_str_idx(extension):
    unicode_sentences = pickle.load(open('preprocessed_' + extension + '.pkl', 'rb'))
    #unicode_sentences is an array of sentences, each represented as an array of byte string words
    str_idx_sentences = []
    unicode2idx = {}
    for u_sent in unicode_sentences:
        str_idx_sent = []
        for u_word in u_sent:
            if u_word not in unicode2idx:
                idx = len(unicode2idx)
                unicode2idx[u_word] = idx
            str_idx_sent.append(str(unicode2idx[u_word]))
        str_idx_sentences.append(str_idx_sent)
    #str_idx_sentences is an array of sentences, each represented as an array of indices
    #unicode2idx is a dictionary mapping each byte string word to its index
    print('Num of sentences:', len(str_idx_sentences))
    print('Check sentence0:', str_idx_sentences[0])
    pickle.dump(str_idx_sentences, open('sentences_' + extension + '.pkl', 'wb'))
    pickle.dump(unicode2idx, open('unicode2idx_' + extension + '.pkl', 'wb'))

def train_embeddings(extension):
    sentences = pickle.load(open('sentences_' + extension + '.pkl', 'rb'))
    # our w2v settings for our dataset
    size = 300
    window = 5
    min_count = 5
    epochs = 20
    print('Training Word2Vec...')
    model = Word2Vec(sentences, size=size, window=window,
					 min_count=min_count, iter=epochs)
    wv = model.wv
    print('Finished. Vocab size:', len(wv.vocab))
    vocab = list(sorted([w for w in wv.vocab], key=lambda x: int(x)))  # sort by idx
    print('First 10 words in vocab:', vocab[:10])
    print('Last 10 words in vocab:', vocab[-10:])

    out_file = '../data/{0}_w2v_s{1}_w{2}_mc{3}_ep{4}.bin'.format(extension, size, window, min_count, epochs)
    wv.save_word2vec_format(out_file, binary=True)
    print('Word2Vec vectors saved to', out_file)

def make_word_emb_for_nn(extension):
    size = 300
    window = 5
    min_count = 5
    epochs = 20
    w2v_file = '../data/{0}_w2v_s{1}_w{2}_mc{3}_ep{4}.bin'.format(extension, size, window, min_count, epochs)
    wv = KeyedVectors.load_word2vec_format(w2v_file, binary=True)
    print('Number of embeddings in {}: {}'.format(w2v_file, len(wv.vocab)))

    unicode2idx_pkl = 'unicode2idx_' + extension + '.pkl'
    unicode2idx = pickle.load(open(unicode2idx_pkl, 'rb'))  # complete vocab
    print('Size of complete vocab:', len(unicode2idx))

    dl = Data_loader(labeled_only=True)
    vocab_size = 40000
    dim = 300
    embeds = np.zeros((vocab_size, dim), dtype=np.float)
    embeds[1] = np.random.uniform(-0.25, 0.25, dim)
    not_in_vocab = 0
    not_in_w2v = 0
    unknown_idx = set()
    avg_vocab = np.zeros(dim)
    known_vocab = 0
    for dl_idx in range(2, vocab_size):
        unicode = dl.convert2unicode([dl_idx]).encode('utf-8')
        if unicode in unicode2idx:
            ext_idx = unicode2idx[unicode]
            if str(ext_idx) in wv.vocab:
                known_vocab += 1
                embeds[dl_idx] = wv[str(ext_idx)]
                avg_vocab += wv[str(ext_idx)]
            else:
                #this word is in the training corpus of the pretrained embedding but is thrown away
                #because its frequency does not reach min_count = 5
                not_in_w2v += 1
                unknown_idx.add(dl_idx)
                #embeds[dl_idx] = np.random.uniform(-0.25, 0.25, dim)
        else:
            #this word is not even in the training corpus of the pretrained embedding
            not_in_vocab += 1
            unknown_idx.add(dl_idx)
            #embeds[dl_idx] = np.random.uniform(-0.25, 0.25, dim)

    #assign unknown vocabs to average of known vocabs
    avg_vocab /= known_vocab
    for unk_idx in unknown_idx:
        embeds[unk_idx] = avg_vocab

    print(not_in_vocab, 'not in vocab')
    print(not_in_w2v, 'not in word2vec (min_count=5)')
    missed = not_in_vocab + not_in_w2v
    print('Total: got {} embeddings, missed {}, out of {}'.format(vocab_size-missed, missed, vocab_size))

    save_file = 'word_emb_' + extension + '.np'
    np.savetxt(save_file, embeds) #embeds is final embedding by idx
    print('Saved embeddings in', save_file)


def check_our_stats():
    wv = KeyedVectors.load_word2vec_format('../data/w2v_word_s300_w5_mc5_ep20.bin', binary=True)
    print('Number of embeddings in our W2V:', len(wv.vocab))

    splex = pickle.load(open('../data/splex_standard_svd_word_s300_seeds_hc.pkl', 'rb'))
    print('Number of embeddings in SPLex:', len(splex))

    in_w2v = 2  # everyone gets 0 and 1 as freebies
    in_splex = 2
    vocab_size = 40000
    for idx in range(2, vocab_size):
        str_idx = str(idx)
        if str_idx in wv.vocab:
            in_w2v += 1
        if str_idx in splex:
            in_splex += 1
    print('Our W2V: got {} embeddings, missed {}'.format(in_w2v, vocab_size-in_w2v))
    print('SPLex: got {} embeddings, missed {}'.format(in_splex, vocab_size-in_splex))


def train_aave():
    # aave embedding is trained with tweets using Word2Vec in this file.
    preprocess_aave()
    # twitter number of sentences 1148141
    print("preprocess done")
    unicode_to_str_idx('aave')
    print("unicode to string done")
    train_embeddings('aave')
    # twitter-data derived valid vocab size 37854 (reach min-count)
    # twitter-data derived all vocab size 307095
    print("train embeddings done")
    make_word_emb_for_nn('aave')
    # 13382 of the 40000 words are not in vocab of 307095 size
    # 6734 of the 40000 words are not in word2vec of 37854 size
    # got 19884 embeddings, missed 20116, out of 40000
    # all the UNKs (either not in vocab or not in word2vec) are represented using random real from -0.25 to 0.25
    print("make word emb for nn done")
    # We trained a parallel set of word embeddings on the African American English (AAE) corpus of around 1.1 million
    # tweets provided by Blodgett et al. 2016
    # and another set on a corpus of a location-specific set of tweets that we scraped, drawn from users who posted from
    # a specific area within the South Side of Chicago where the gangs we study are based.
    # We also compared performance with a randomly initialized word em- bedding matrix.


def train_chicago():
    # aave embedding is trained with tweets using Word2Vec in this file.
    preprocess_chicago()
    # twitter number of sentences 813572
    print("preprocess done")
    unicode_to_str_idx('chicago')
    print("unicode to string done")
    train_embeddings('chicago')
    # twitter-data derived valid vocab size 33934 (reach min-count)
    # twitter-data derived all vocab size 92588
    print("train embeddings done")
    make_word_emb_for_nn('chicago')
    # 17867 of the 40000 words are not in vocab of 307095 size
    # 5171 of the 40000 words are not in word2vec of 37854 size
    # got 16962 embeddings, missed 23038, out of 40000
    # all the UNKs (either not in vocab or not in word2vec) are represented using random real from -0.25 to 0.25
    print("make word emb for nn done")
    # We trained a parallel set of word embeddings on the African American English (AAE) corpus of around 1.1 million
    # tweets provided by Blodgett et al. 2016
    # and another set on a corpus of a location-specific set of tweets that we scraped, drawn from users who posted from
    # a specific area within the South Side of Chicago where the gangs we study are based.
    # We also compared performance with a randomly initialized word em- bedding matrix.

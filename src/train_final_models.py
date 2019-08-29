import pickle as pkl
from nn_experiment import Experiment
import sys
import numpy as np
from dropout_reg import Word_Dropper

def train_model(model_id, run_idx):
    input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))

    #1. CNN + DS Word Embedding + Context Features
    if model_id == 1:
        input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
        pretrained_weight_dirs = {'aggression_word_embed': ['../weights_average/word_emb_w2v.np'],
                                  'loss_word_embed': ['../weights_average/word_emb_w2v.np']}
        options = ['word']
        experiment = Experiment(mode='standard', experiment_dir='CNN+DSWEMB+CF_run_%d' % run_idx,
                                pretrained_weight_dirs=pretrained_weight_dirs,
                                options=options,
                                input_name2id2np=input_name2id2np)
        experiment.cv()

    #2. CNN + Twitter Word Embedding + Context Features
    if model_id == 2:
        input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
        pretrained_weight_dirs = {'aggression_word_embed': ['../weights_average/word_emb_twitter.np'],
                                'loss_word_embed': ['../weights_average/word_emb_twitter.np']}
        options = ['word']
        experiment = Experiment(mode='standard', experiment_dir='CNN+TwitterWEMB+CF_run_%d' % run_idx,
                                pretrained_weight_dirs=pretrained_weight_dirs,
                                options=options, embed_dim=200,
                                input_name2id2np=input_name2id2np)
        experiment.cv()

    #3. CNN + DS Word Embedding + Context Features + DS ELMo
    if model_id == 3:
        input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
        pretrained_weight_dirs = {'aggression_word_embed': ['../weights_average/word_emb_w2v_256.np'],
                                'loss_word_embed': ['../weights_average/word_emb_w2v_256.np']}
        options = ['word']
        experiment = Experiment(mode='standard', experiment_dir='CNN+DSWEMB+CF+DSELMo_run_%d' % run_idx,
                                pretrained_weight_dirs=pretrained_weight_dirs,
                                options=options, input_format="both", embed_dim=256,
                                input_name2id2np=input_name2id2np, elmo_representation_dir="../data/DS_ELMo_rep")
        experiment.cv()

    #4. CNN + DS Word Embedding + DS ELMo
    if model_id == 4:
        input_name2id2np = None
        pretrained_weight_dirs = {'aggression_word_embed': ['../weights_average/word_emb_w2v_256.np'],
                                  'loss_word_embed': ['../weights_average/word_emb_w2v_256.np']}
        options = ['word']
        experiment = Experiment(mode='standard', experiment_dir='CNN+DSWEMB+DSELMo_run_%d' % run_idx,
                                pretrained_weight_dirs=pretrained_weight_dirs,
                                options=options, input_format="both", embed_dim=256,
                                input_name2id2np=input_name2id2np, elmo_representation_dir="../data/DS_ELMo_rep")
        experiment.cv()

    #5. CNN + DS Word Embedding + Context Features + Non-DS ELMo
    if model_id == 5:
        input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
        pretrained_weight_dirs = {'aggression_word_embed': ['../weights_average/word_emb_w2v_1024.np'],
                                  'loss_word_embed': ['../weights_average/word_emb_w2v_1024.np']}
        options = ['word']
        experiment = Experiment(mode='standard', experiment_dir='CNN+DSWEMB+CF+NonDSELMo_run_%d' % run_idx,
                                pretrained_weight_dirs=pretrained_weight_dirs,
                                options=options, input_format="both", embed_dim=1024,
                                input_name2id2np=input_name2id2np, elmo_dim=1024, elmo_representation_dir="../data/NonDS_ELMo_rep")
        experiment.cv()

    #6. LSTM Attention + DS Word Embedding + DS ELMo + Context Features
    if model_id == 6:
        input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
        experiment = Experiment(mode='lstm_attention', experiment_dir='LSTMAttn+DSWEMB+DSELMo+CF_run_%d' % run_idx,
                                epochs=10, input_format="both", embed_dim=256,
                                word_embedding_matrix=np.loadtxt('../weights_average/word_emb_w2v_256.np'),
                                input_name2id2np=input_name2id2np, elmo_representation_dir="../data/DS_ELMo_rep")
        experiment.cv()

    #7. LSTM Attention + DS Word Embedding + DS ELMo + Context Features + Rationale
    if model_id == 7:
        input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
        experiment = Experiment(mode='lstm_attention', experiment_dir='LSTMAttn+DSWEMB+DSELMo+CF+RTL_run_%d' % run_idx,
                                epochs=10, input_format="both", embed_dim=256,
                                word_embedding_matrix=np.loadtxt('../weights_average/word_emb_w2v_256.np'),
                                input_name2id2np=input_name2id2np, elmo_representation_dir="../data/DS_ELMo_rep",
                                use_rationale=True)
        experiment.cv()

    #8. LSTM Attention + DS Word Embedding + Context Features
    if model_id == 8:
        input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
        experiment = Experiment(mode='lstm_attention', experiment_dir='LSTMAttn+DSWEMB+CF_run_%d' % run_idx,
                                epochs=10, word_embedding_matrix=np.loadtxt('../weights_average/word_emb_w2v.np'),
                                input_name2id2np=input_name2id2np)
        experiment.cv()

    #9. LSTM Attention + DS Word Embedding + Context Features + Rationale
    if model_id == 9:
        input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
        experiment = Experiment(mode='lstm_attention', experiment_dir='LSTMAttn+DSWEMB+CF+RTL_run_%d' % run_idx,
                                epochs=10, word_embedding_matrix=np.loadtxt('../weights_average/word_emb_w2v.np'),
                                input_name2id2np=input_name2id2np, use_rationale=True)
        experiment.cv()

    #13. LSTM Attention + DS Word Embedding + CF + KRdropout
    if model_id == 13:
        input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
        wd = Word_Dropper(dropout_prob=0.5, mode='keep_r')
        experiment = Experiment(mode='lstm_attention', experiment_dir='LSTMAttn+DSWEMB+CF+KR_run_%d' % run_idx,
                                epochs=30, word_embedding_matrix=np.loadtxt('../weights_average/word_emb_w2v.np'),
                                input_name2id2np=input_name2id2np,
                                use_generator=True, word_dropper=wd)
        experiment.cv()

    #14. LSTM Attention + DS Word Embedding + CF + CY
    if model_id == 14:
        input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
        wd = Word_Dropper(dropout_prob=0.5, mode='change_y')
        experiment = Experiment(mode='lstm_attention', experiment_dir='LSTMAttn+DSWEMB+CF+CY_run_%d' % run_idx,
                                min_epochs=15, epochs=30, word_embedding_matrix=np.loadtxt('../weights_average/word_emb_w2v.np'),
                                input_name2id2np=input_name2id2np,
                                use_generator=True, word_dropper=wd)
        experiment.cv()

    #15. LSTM Attention + DS Word Embedding + CF + DROP
    if model_id == 15:
        input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
        wd = Word_Dropper(dropout_prob=0.5, mode=None)
        experiment = Experiment(mode='lstm_attention', experiment_dir='LSTMAttn+DSWEMB+CF+DROP_run_%d' % run_idx,
                                epochs=30, word_embedding_matrix=np.loadtxt('../weights_average/word_emb_w2v.np'),
                                input_name2id2np=input_name2id2np,
                                use_generator=True, word_dropper=wd)
        experiment.cv()


    #18. LSTM Attention + DS Word Embedding + CF at last
    if model_id == 18:
        input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
        experiment = Experiment(mode='lstm_attention', experiment_dir='LSTMAttn+DSWEMB+CFLAST_run_%d' % run_idx,
                                epochs=10, word_embedding_matrix=np.loadtxt('../weights_average/word_emb_w2v.np'),
                                input_name2id2np=input_name2id2np, context_features_before=False)
        experiment.cv()

    #19. LSTM Attention + DS Word Embedding + CF at last + Rationale
    if model_id == 19:
        input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
        experiment = Experiment(mode='lstm_attention', experiment_dir='LSTMAttn+DSWEMB+CFLAST+RTL_run_%d' % run_idx,
                                epochs=10, word_embedding_matrix=np.loadtxt('../weights_average/word_emb_w2v.np'),
                                input_name2id2np=input_name2id2np, context_features_before=False, use_rationale=True)
        experiment.cv()

if __name__ == '__main__':
    model_id = int(sys.argv[1])
    for run_idx in range(5):
        train_model(model_id, run_idx)

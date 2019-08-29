import pickle as pkl
from nn_experiment import extract_dim_input_name2id2np
from model_def import NN_architecture
from model import Model
from LSTM_attn import Attn_LSTM
from model_info import model_ids

id2model_dir = model_ids()

def load_model(model_id, run_idx, fold_idx, class_idx):
    input_name2id2np = pkl.load(open('../data/emnlp2018_best.pkl', 'rb'))
    input_dim_map = extract_dim_input_name2id2np(input_name2id2np)

    #1. CNN + DS Word Embedding + Context Features
    if model_id == 1:
        model = NN_architecture(input_dim_map=input_dim_map).model
        model.load_weights("../experiments/CNN+DSWEMB+CF_run_%d/%d_%d.weight" % (run_idx, fold_idx, class_idx))
        return model

    #2. CNN + Twitter Word Embedding + Context Features
    if model_id == 2:
        model = NN_architecture(input_dim_map=input_dim_map, embed_dim=200).model
        model.load_weights("../experiments/CNN+TwitterWEMB+CF_run_%d/%d_%d.weight" % (run_idx, fold_idx, class_idx))
        return model

    #3. CNN + DS Word Embedding + Context Features + DS ELMo
    if model_id == 3:
        model = NN_architecture(input_dim_map=input_dim_map, embed_dim=256, input_format="both").model
        model.load_weights("../experiments/CNN+DSWEMB+CF+DSELMo_run_%d/%d_%d.weight" % (run_idx, fold_idx, class_idx))
        return model

    #4. CNN + DS Word Embedding + DS ELMo
    if model_id == 4:
        model = NN_architecture(input_dim_map=None, embed_dim=256, input_format="both").model
        model.load_weights("../experiments/CNN+DSWEMB+DSELMo_run_%d/%d_%d.weight" % (run_idx, fold_idx, class_idx))
        return model

    #5. CNN + DS Word Embedding + Context Features + Non-DS ELMo
    if model_id == 5:
        model = NN_architecture(input_dim_map=input_dim_map, embed_dim=1024, input_format="both", elmo_dim=1024).model
        model.load_weights("../experiments/CNN+DSWEMB+CF+NonDSELMo_run_%d/%d_%d.weight" % (run_idx, fold_idx, class_idx))
        return model

    #6. LSTM Attention + DS Word Embedding + DS ELMo + Context Features
    if model_id == 6:
        model = Attn_LSTM(input_dim_map=input_dim_map, input_format="both", embedding_dim=256, vocab_size=40000)
        model_wrapper = Model(model, mode='eval', weight_dir="../experiments/LSTMAttn+DSWEMB+DSELMo+CF_run_%d/%d_%d.weight"
                                                             % (run_idx, fold_idx, class_idx))
        return model_wrapper

    #7. LSTM Attention + DS Word Embedding + DS ELMo + Context Features + Rationale
    if model_id == 7:
        model = Attn_LSTM(input_dim_map=input_dim_map, input_format="both", embedding_dim=256, vocab_size=40000)
        model_wrapper = Model(model, mode='eval', weight_dir="../experiments/LSTMAttn+DSWEMB+DSELMo+CF+RTL_run_%d/%d_%d.weight"
                                                             % (run_idx, fold_idx, class_idx))
        return model_wrapper

    #8. LSTM Attention + DS Word Embedding + Context Features
    if model_id == 8:
        model = Attn_LSTM(input_dim_map=input_dim_map, input_format="discrete", embedding_dim=300, vocab_size=40000)
        model_wrapper = Model(model, mode='eval', weight_dir="../experiments/LSTMAttn+DSWEMB+CF_run_%d/%d_%d.weight"
                                                            % (run_idx, fold_idx, class_idx))
        return model_wrapper

    #9. LSTM Attention + DS Word Embedding + Context Features + Rationale
    if model_id == 9:
        model = Attn_LSTM(input_dim_map=input_dim_map, input_format="discrete", embedding_dim=300, vocab_size=40000)
        model_wrapper = Model(model, mode='eval', weight_dir="../experiments/LSTMAttn+DSWEMB+CF+RTL_run_%d/%d_%d.weight"
                                                             % (run_idx, fold_idx, class_idx))
        return model_wrapper

    #10. CNN + DS Word Embedding + Context Features + KRdropout
    #11. CNN + DS Word Embedding + Context Features + CY
    #12. CNN + DS Word Embedding + Context Features + DROP
    if model_id in [10, 11, 12]:
        model = NN_architecture(input_dim_map=input_dim_map).model
        model.load_weights("%s%d/%d_%d.weight" % (id2model_dir[model_id], run_idx, fold_idx, class_idx))
        return model

    #13. LSTM Attention + DS Word Embedding + CF + KRdropout
    #14. LSTM Attention + DS Word Embedding + CF + CY
    #15. LSTM Attention + DS Word Embedding + CF + DROP
    if model_id in [13, 14, 15]:
        model = Attn_LSTM(input_dim_map=input_dim_map, input_format="discrete", embedding_dim=300, vocab_size=40000)
        model_wrapper = Model(model, mode='eval', weight_dir="%s%d/%d_%d.weight" % (id2model_dir[model_id], run_idx, fold_idx, class_idx))
        return model_wrapper

    #16. LSTM Attention + DS Word Embedding
    if model_id == 16:
        model = Attn_LSTM(input_dim_map=None, input_format="discrete", embedding_dim=300, vocab_size=40000,
                          context_features_before=False, context_features_last=False)
        model_wrapper = Model(model, mode='eval', weight_dir="%s%d/%d_%d.weight" % (id2model_dir[model_id], run_idx, fold_idx, class_idx))
        return model_wrapper

    #17. LSTM Attention + DS Word Embedding + Rationale
    if model_id == 17:
        model = Attn_LSTM(input_dim_map=None, input_format="discrete", embedding_dim=300, vocab_size=40000,
                          context_features_before=False, context_features_last=False)
        model_wrapper = Model(model, mode='eval', weight_dir="%s%d/%d_%d.weight" % (id2model_dir[model_id], run_idx, fold_idx, class_idx))
        return model_wrapper

    #18. LSTM Attention + DS Word Embedding + CF at last
    #19. LSTM Attention + DS Word Embedding + CF at last + Rationale
    if model_id in [18, 19]:
        model = Attn_LSTM(input_dim_map=input_dim_map, input_format="discrete", embedding_dim=300, vocab_size=40000, context_features_before=False)
        model_wrapper = Model(model, mode='eval', weight_dir="%s%d/%d_%d.weight" % (id2model_dir[model_id], run_idx, fold_idx, class_idx))
        return model_wrapper

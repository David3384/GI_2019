import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from util import trim

torch.set_num_threads(1)

# the lstm_attention architecture
class Attn_LSTM(nn.Module):

    def __init__(self, input_format="discrete", prefix="general", use_attn=True, context_features_before=True,
                 context_features_last=True, hidden_dim=64, dropout_lstm=0.5, input_dim_map=None, verbose=False, #general parameters
                 word_embedding_matrix=None, embedding_dim=None, vocab_size=None, dropout_embedding=0.5, #discrete input format parameters
                 elmo_depth=3, elmo_dim=256, elmo_option='merge_average', elmo_dropout=0, #elmo input format parameters
                 **kwargs
                 ):
        super(Attn_LSTM, self).__init__()

        # whether the model uses attention mechanism
        self.use_attn = use_attn

        # whether the model uses context features before applying attention, after applying attention or does not use
        # context features
        self.context_feature_before = context_features_before
        self.context_feature_last = context_features_last

        self.input_format = input_format

        if input_dim_map is None:
            self.input_dim_map = {}
        else:
            self.input_dim_map = input_dim_map

        # either uses randomization or load embedding matrix (xor)
        if self.input_format == 'discrete':
            assert bool(vocab_size) != (word_embedding_matrix is not None)
            assert bool(vocab_size) == bool(embedding_dim)
        else:
            self.elmo_depth, self.elmo_dim, self.elmo_option, self.dropout_elmo = elmo_depth, elmo_dim, elmo_option, elmo_dropout

        self.verbose = verbose # print debugging message

        assert self.input_format in ['discrete', 'elmo', 'both']
        if self.input_format == 'discrete' or self.input_format == 'both':
            if vocab_size is None:
                self.wemb = Embedding.from_pretrained(torch.from_numpy(word_embedding_matrix),
                                                      freeze=False)
                vocab_size, embedding_dim = word_embedding_matrix.shape
            else:
                self.wemb = Embedding(vocab_size, embedding_dim)
            # embedding dropout layer
            self.dropout_emb = nn.Dropout(dropout_embedding)

        if self.input_format == 'elmo':
            self.kernel_weight = nn.Parameter(torch.randn(elmo_depth, 1), requires_grad=True)
            if self.elmo_option == 'add_average':
                self.second_kernel_weight = nn.Parameter(torch.randn(elmo_depth + 1, 1), requires_grad=True)
            self.dropout_elmo = nn.Dropout(elmo_dropout)

        if self.input_format == 'both':
            self.kernel_weight = nn.Parameter(torch.randn(elmo_depth + 1, 1), requires_grad=True)
            if self.elmo_option == 'add_average':
                self.second_kernel_weight = nn.Parameter(torch.randn(elmo_depth + 2, 1), requires_grad=True)
            self.dropout_elmo = nn.Dropout(elmo_dropout)

        # print model parameters
        self.property = []
        self.property.append("lstm %s" % prefix)
        self.property.append("use attention: %s, context features before: %s, context features after: %s" % (
        self.use_attn, self.context_feature_before, self.context_feature_last))
        self.property.append("hidden_dim: %d, dropout_lstm: %.1f, import input_dim_map: %d" % (
        hidden_dim, dropout_lstm, self.input_dim_map != {}))
        if input_format == "discrete" or input_format == 'both':
            if word_embedding_matrix is not None:
                self.property.append("embedding loaded successfully.")
            else:
                self.property.append("randomly initialized embedding.")
            self.property.append('embedding dropout: %.1f, vocab size: %d' % (dropout_embedding, vocab_size))
        elif input_format == 'elmo' or input_format == 'both':
            self.property.append('elmo depth: %d, elmo_dim: %d, elmo_option: %s, elmo dropout: %.1f' % (
            self.elmo_depth, self.elmo_dim, self.elmo_option, elmo_dropout))

        # lstm dropout layer
        self.dropout_lstm = nn.Dropout(dropout_lstm)

        # initialize encoder (LSTM)
        if self.input_format == 'discrete':
            self.encoder = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        else:
            self.encoder = nn.LSTM(elmo_dim, hidden_dim, bidirectional=True)

        # weights and layers
        hidden_dim = hidden_dim * 2
        total_input_dimension = hidden_dim

        self.context_feature_names = None
        self.context_feature_dimensions = None
        if self.context_feature_before:
            context_feature_dimensions, context_feature_names = self.get_context_features_info()
            self.context_feature_dimensions = context_feature_dimensions
            self.context_feature_names = context_feature_names
            total_input_dimension += context_feature_dimensions

        if self.use_attn:
            self.W_a = nn.Linear(total_input_dimension, 1)

        if self.context_feature_last:
            if self.context_feature_dimensions is not None:
                total_input_dimension += self.context_feature_dimensions
            else:
                context_feature_dimensions, context_feature_names = self.get_context_features_info()
                self.context_feature_dimensions = context_feature_dimensions
                self.context_feature_names = context_feature_names
                total_input_dimension += context_feature_dimensions

        self.fc = nn.Linear(total_input_dimension, 1)


    def get_context_features_info(self):
        """
        Get the context feature names and the total dimension of all context features concatenated
        """
        total_context_features_dimension = 0
        context_feature_names = []
        for input_name in self.input_dim_map:
            dim = self.input_dim_map[input_name]
            total_context_features_dimension += dim
            context_feature_names.append(input_name + '_input')
        return total_context_features_dimension, sorted(context_feature_names)


    # forward propagation
    # X['word_content_input'] is a list of integers (sequence of tokens indexed)
    # X[context_feature_name] is the user's feature for a context feature
    # takes input sentence, context_features, returns output 'output' and 'attn'
    def forward(self, X):
        if self.input_format == 'discrete':
            sentence = Variable(torch.LongTensor(trim(X['word_content_input'])))
            embedded_val = self.dropout_emb(self.wemb(sentence[:, None]))
        else:
            #permute axis order for convenience of model forward
            #Elmo reorder of dimension axis is performed by the model itself.
            elmo_rep_permuted = np.swapaxes(np.swapaxes(X['word_content_input_elmo'], 0, 2), 0, 1)

        if self.input_format == "elmo":
            self.normalized_weight = F.softmax(self.kernel_weight, dim=0)
            if self.elmo_option == 'add_average':
                self.second_normalized_weight = F.softmax(self.second_kernel_weight, dim=0)

            sentence = Variable(torch.FloatTensor(elmo_rep_permuted))

            average_repr = torch.squeeze(sentence @ self.normalized_weight, dim=2)
            if self.elmo_option == 'add_average':
                all_repr = torch.cat((sentence, average_repr.unsqueeze(dim=2)), dim=2)
                average_repr = torch.squeeze(all_repr @ self.second_normalized_weight, dim=2)
            average_repr = torch.unsqueeze(average_repr, dim=1)
            embedded_val = self.dropout_elmo(average_repr)

        elif self.input_format == 'both':
            self.normalized_weight = F.softmax(self.kernel_weight, dim=0)
            if self.elmo_option == 'add_average':
                self.second_normalized_weight = F.softmax(self.second_kernel_weight, dim=0)

            #discrete input format features
            sentence = Variable(torch.LongTensor(trim(X['word_content_input'])))
            discrete_embedded_val = self.dropout_emb(self.wemb(sentence[:, None])).permute(0, 2, 1)

            #elmo input format features
            sentence = Variable(torch.FloatTensor(elmo_rep_permuted))
            sentence = torch.cat((discrete_embedded_val, sentence), dim=2)
            average_repr = torch.squeeze(sentence @ self.normalized_weight, dim=2)
            if self.elmo_option == 'add_average':
                all_repr = torch.cat((sentence, average_repr.unsqueeze(dim=2)), dim=2)
                average_repr = torch.squeeze(all_repr @ self.second_normalized_weight, dim=2)
            average_repr = torch.unsqueeze(average_repr, dim=1)
            embedded_val = self.dropout_elmo(average_repr)

        hs, (q, _) = self.encoder(embedded_val.float())

        if self.use_attn:
            hs = self.dropout_lstm(hs)
        else:
            q = self.dropout_lstm(q)
        hs = hs.squeeze_(dim=1)

        # get concatenated representation of hidden states with/without context features
        concatenated_representation = []
        for hidden_state in hs:
            if self.context_feature_before:
                for feature_name in self.context_feature_names:
                    hidden_state = torch.cat((hidden_state, torch.Tensor(X[feature_name])))
                concatenated_representation.append(hidden_state)
            else:
                concatenated_representation.append(hidden_state)
        concatenated_representation = torch.stack(concatenated_representation)

        if self.verbose:
            print("concatenated hidden states' shape: " + str(concatenated_representation.shape))

        # get concatenated representation of last state of LSTM with/without context featuress
        concatenated_q = q.view(1, -1)[0]
        if self.context_feature_before:
            for feature_name in self.context_feature_names:
                concatenated_q = torch.cat((concatenated_q, torch.Tensor(X[feature_name])))

        if self.verbose:
            print("concatenated last state's shape: " + str(concatenated_q.shape))

        # calculating attntion
        if not self.use_attn:
            z = concatenated_q
        else:
            A = F.softmax(torch.tanh(self.W_a(concatenated_representation)), dim=0)
            z = (torch.transpose(A, 0, 1) @ concatenated_representation)[0]

        if self.verbose:
            print("feature representation shape: " + str(z.shape))

        if self.context_feature_last:
            for feature_name in self.context_feature_names:
                z = torch.cat((z, torch.Tensor(X[feature_name])))

        if self.verbose:
            print("last tensor representation shape: " + str(z.shape))

        output = torch.sigmoid(self.fc(z)).view(-1, )

        return_dict = {'output': output}
        if self.use_attn:
            return_dict['attn'] = A.view(1, -1)

        return return_dict


# use this function to define a basic model used for debugging
def _prelim_model():
    word_embedding_matrix = np.loadtxt('../weights_average/word_emb_w2v.np')
    model = Attn_LSTM(hidden_dim=64, word_embedding_matrix=word_embedding_matrix,
                      use_attn=True, context_features_before=True, context_features_last=True,
                      input_dim_map={'splex_score': 3, 'cl_score': 2}, verbose=True, input_format="elmo", elmo_option="add_average")
    return model


if __name__ == '__main__':
    model = _prelim_model()
    X = {'word_content_input':[0,124,51,21,25,12,42], 'splex_score_input':[35,12,61], 'cl_score_input':[64,19]}
    #X = {'word_content_input':np.ones((50, 256, 3)), 'splex_score_input':[35,12,61], 'cl_score_input':[64,19]}
    return_dict = model.eval()(X)
    print(return_dict)

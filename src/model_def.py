import numpy as np
from keras.layers import Input, Dense, Conv1D, Embedding, concatenate, \
    GlobalMaxPooling1D, Dropout, Flatten, Lambda
from keras.models import Model
from keras import backend as K
from keras.layers import Layer
from tensorflow import tensordot, concat


# returns two tensors
# one for input_content, the other for tensor before final classification
def content2rep(option='word', vocab_size=40000, max_len=50, drop_out=0.5,
                filter=200, dense_size=256, embed_dim=300,
                kernel_range=(1,3), prefix='general', input_format='discrete',
                elmo_depth=3, elmo_dim=256, elmo_option='merge_average', elmo_dropout=0):

    input_contents = []
    if input_format == 'discrete':
        # input layer
        # input will not have a prefix in its name
        input_content = Input(shape=(max_len,),
                              name= option + '_content_input')
        # embedding layer
        embed_layer = Embedding(vocab_size, embed_dim, input_length=max_len,
                                name= prefix + '_' + option + '_embed')
        e_i = embed_layer(input_content)
        embed_drop_out = Dropout(drop_out, name=prefix + '_' + option + '_embed_dropout')
        input_contents.append(input_content)

    elif input_format == 'elmo':
        input_content = Input(shape=(elmo_depth, max_len, elmo_dim),
                              name= option + '_content_input_elmo')
        #Elmo reorder of dimension axis is performed by the model itself.
        reordered_input_content = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)))(input_content) #permute to (max_len, elmo_dim, elmo_depth)
        e_i = ELMo_combine(output_dim=(max_len, elmo_dim), elmo_layers=elmo_depth, option=elmo_option)(reordered_input_content)
        embed_drop_out = Dropout(elmo_dropout, name=prefix + '_' + option + '_elmo_dropout')
        input_contents.append(input_content)

    elif input_format == 'both':
        #discrete input
        input_content = Input(shape=(max_len,), name=option + '_content_input')
        embed_layer = Embedding(vocab_size, embed_dim, input_length=max_len, name=prefix + '_' + option + '_embed')
        e_i = embed_layer(input_content)
        embed_drop_out = Dropout(drop_out, name=prefix + '_' + option + '_embed_dropout')
        discrete_repr = Lambda(lambda x: K.expand_dims(x))(embed_drop_out(e_i))

        #elmo input
        elmo_input_content = Input(shape=(elmo_depth, max_len, elmo_dim), name=option + '_content_input_elmo')
        #Elmo reorder of dimension axis is performed by the model itself.
        reordered_elmo_input_content = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)))(elmo_input_content) #permute to (max_len, elmo_dim, elmo_depth)
        all_input_content = concatenate([discrete_repr, reordered_elmo_input_content])
        e_i = ELMo_combine(output_dim=(max_len, elmo_dim), elmo_layers=elmo_depth + 1, option=elmo_option)(all_input_content)
        embed_drop_out = Dropout(elmo_dropout, name=prefix + '_' + option + '_elmo_dropout')
        input_contents = [input_content, elmo_input_content]

    e_i = embed_drop_out(e_i)

    # convolutional layers
    conv_out = []
    for kernel_size in kernel_range:
        c = Conv1D(filter, kernel_size, activation='relu', name= prefix + '_' + option + '_conv_' + str(kernel_size))(e_i)
        c = GlobalMaxPooling1D(name= prefix + '_' + option + '_max_pooling_' + str(kernel_size))(c)
        c = Dropout(drop_out, name= prefix + '_' + option + '_drop_out_' + str(kernel_size))(c)
        conv_out.append(c)
    agg = concatenate(conv_out)
    dense_layer = Dense(dense_size, activation='relu',
                        name= prefix + '_' + option + '_last')
    content_rep = dense_layer(agg)
    return input_contents, content_rep


class ELMo_combine(Layer):
    # Class to combine multiple layers of elmo representation into single layer using weighted sum (weights are trainable)
    def __init__(self, output_dim, elmo_layers, option="merge_average", **kwargs):
        """
        merge_option can take "merge_average" or "add_average".
        "merge_average" means concatenate the L layers of ELMo representations to a weighted sum to get one layer.
        "add_average" means first get a weighted sum of the L ELMo layer representations, and get a new weighted
        sum of the (L + 1) ELMo layer representations to get one layer.
        """
        self.output_dim = output_dim
        self.merge_option = option
        self.elmo_layers = elmo_layers
        super(ELMo_combine, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.elmo_layers,), #elmo_depth
                                      initializer='uniform',
                                      trainable=True)
        self.weighted_kernel = K.softmax(self.kernel)

        if self.merge_option == "add_average":
            self.second_kernel = self.add_weight(name='second_kernel',
                                                 shape=(self.elmo_layers + 1,),
                                                 initializer='uniform',
                                                 trainable=True)
            self.second_weighted_kernel = K.softmax(self.second_kernel)
        super(ELMo_combine, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        result = tensordot(x, self.weighted_kernel, axes=([3], [0]))
        if self.merge_option == "merge_average":
            return result
        else:
            all_representation = concat([x, K.expand_dims(result, -1)], axis=-1)
            return tensordot(all_representation, self.second_weighted_kernel, axes=([3], [0]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.output_dim)


# return a boolean checking whether an input name is in user index format
def input_name_is_user_idx(input_name):
    if (('usr' in input_name or 'user' in input_name) # capturing various personal spelling habbits to prevent bug
        and ('idx' in input_name or 'index' in input_name or 'idex' in input_name)):
        return True
    return False

class NN_architecture:

    def __init__(self,
                 options=['word'],
                 input_dim_map=None,
                 word_vocab_size=40000, word_max_len=50,
                 char_vocab_size=1200, char_max_len=150,
                 drop_out=0.5,
                 filter=200, dense_size=256, embed_dim=300, kernel_range=range(1,3),
                 pretrained_weight_dirs=None, weight_in_keras=None,
                 num_users=50, user_embed_dim=32, user_embed_dropout=0,
                 interaction_layer_dim=-1, interaction_layer_drop_out=0.5,
                 mode='cascade',
                 prefix='general',
                 input_format='discrete',
                 elmo_depth=3, elmo_dim=256, elmo_option='merge_average', elmo_dropout=0, #parameter for elmo if elmo is used
                 output_feature=False
                 ):
        """
        Initilizing a neural network architecture according to the specification
        access the actual model by self.model

        Parameters
        ----------
        options: an array containing all the options considered in the neural network model ['char', 'word']
                    (probably splex in the future)
                    for each option, the input is mapped to a lower dimension,
                    then the lower dimension representation of each option is concatenated
                    and is followed by the final classification layer
        input_dim_map: a map from additional input name to its dimension
        word_vocab_size: number of word level vocabs to be considered
        word_max_len: number of words in a tweet sentence
        char_vocab_size: number of char level vocabs to be considered
        char_max_len: number of chars in a tweet sentence
        drop_out: dropout rate for regularization
        filter: number of filters for each kernel size
        dense_size: the size of the dense layer following the max pooling layer
        embed_dim: embedding dimension for character and word level
        kernel_range: range of kernel sizes
        pretrained_weight_dirs: a dictionary containing the pretrained weight.
                    e.g. {'char': '../weights/char_ds.weights'} means that the pretrained weight for character level model
                    is in ../weights/char_ds.weights
        weight_in_keras: whether the weight is in Keras
        """
        self.options, self.prefix = options, prefix

        if input_dim_map is None:
            input_dim_map = {}
        self.input_dim_map = input_dim_map

        # changeable hyper parameter
        self.drop_out = drop_out
        self.word_vocab_size, self.word_max_len = word_vocab_size, word_max_len
        self.char_vocab_size, self.char_max_len = char_vocab_size, char_max_len
        self.num_users, self.user_embed_dim, self.user_embed_dropout = num_users, user_embed_dim, user_embed_dropout
        self.interaction_layer_dim, self.interaction_layer_drop_out = interaction_layer_dim, interaction_layer_drop_out
        self.elmo_depth, self.elmo_dim, self.elmo_option, self.elmo_dropout = elmo_depth, elmo_dim, elmo_option, elmo_dropout #elmo parameters if elmo is used

        # hyper parameters that is mostly fixed
        self.filter, self.dense_size, self.embed_dim, self.kernel_range = filter, dense_size, embed_dim, kernel_range

        self.mode = mode
        self.input_format = input_format
        assert self.input_format in ['discrete', 'elmo', 'both']

        # pretrained_weight directory
        self.pretrained_weight_dirs, self.weight_in_keras = pretrained_weight_dirs, weight_in_keras
        if self.pretrained_weight_dirs is None:
            self.pretrained_weight_dirs = {}
        if self.weight_in_keras is None:
            self.weight_in_keras = {}
        self.output_feature = output_feature
        self.create_model()

        #print model information
        self.property = []
        self.property.append("cnn %s" % self.prefix)
        self.property.append("uses context features: %d" % (self.input_dim_map != {}))
        self.property.append("cnn filter: %d, representation dense size: %d, kernel range: %s" % (self.filter, self.dense_size, str(self.kernel_range)))
        if self.input_format == 'discrete':
            self.property.append("uses pretrained word embedding: %d, word embedding dropout: %.1f, embedding dimension: %d" %
                  ((self.pretrained_weight_dirs != {}), self.drop_out, self.embed_dim))
        elif self.input_format == 'elmo':
            self.property.append("elmo depth: %d, elmo dimension: %d, elmo_option: %s, elmo_dropout: %.1f" % (self.elmo_depth,
                  self.elmo_dim, self.elmo_option, self.elmo_dropout))
        elif self.input_format == 'both':
            self.property.append("uses pretrained word embedding: %d, word embedding dropout: %.1f, embedding dimension: %d" %
                  ((self.pretrained_weight_dirs != {}), self.drop_out, self.embed_dim))
            self.property.append("elmo depth: %d, elmo dimension: %d, elmo_option: %s, elmo_dropout: %.1f" % (self.elmo_depth,
                                                                                       self.elmo_dim, self.elmo_option,
                                                                                       self.elmo_dropout))

    def create_model(self):
        # for each option, create computational graph and load weights
        inputs, last_tensors = [], []
        for option in self.options:

            # how to map char input to the last layer
            if option in ['char', 'word']:
                if option == 'char':
                    input_content, content_rep = content2rep(option,
                                                             self.char_vocab_size, self.char_max_len, self.drop_out,
                                                             self.filter, self.dense_size, self.embed_dim, self.kernel_range,
                                                             self.prefix, self.input_format,
                                                             self.elmo_depth, self.elmo_dim, self.elmo_option, self.elmo_dropout)
                elif option == 'word':
                    input_content, content_rep = content2rep(option,
                                                             self.word_vocab_size, self.word_max_len, self.drop_out,
                                                             self.filter, self.dense_size, self.embed_dim, self.kernel_range,
                                                             self.prefix, self.input_format,
                                                             self.elmo_depth, self.elmo_dim, self.elmo_option, self.elmo_dropout)
                inputs.extend(input_content)
                last_tensors.append(content_rep)


        # the user name needs to have "user_idx" suffix to be considered user idx
        need_user_embedding = False
        for input_name in self.input_dim_map:
            if input_name_is_user_idx(input_name):
                need_user_embedding = True
        if need_user_embedding:
            user_embedding = Embedding(self.num_users, self.user_embed_dim, input_length=1,
                                       name=self.prefix + '_user_embed')
            user_embed_dropout_layer = Dropout(self.user_embed_dropout,
                                               name=self.prefix + '_user_embed_dropout')

        # directly concatenate addtional inputs (such as splex scores and context representations)
        # to the last layer
        for input_name in sorted(self.input_dim_map.keys()):
            if input_name_is_user_idx(input_name):
                input = Input(shape=(1,),
                              name=input_name + '_input')
                inputs.append(input)
                # flatten the user embedding (after dropout)
                input_embed = Flatten()(user_embed_dropout_layer(user_embedding(input)))
                last_tensors.append(input_embed)
            else:
                input = Input(shape=(self.input_dim_map[input_name],),
                                      name=input_name + '_input')
                inputs.append(input)
                last_tensors.append(input)

        # concatenate all the representations
        if len(last_tensors) >= 2:
            concatenated_rep = concatenate(last_tensors)
        else:
            concatenated_rep = last_tensors[0]

        # out layer
        if self.mode == 'ternary':
            self.out_layer = Dense(3, activation='softmax',
                                   name=self.prefix + '_classification')
        elif self.mode == 'cascade':
            self.out_layer = Dense(1, activation='sigmoid',
                                  name=self.prefix+ '_classification')
        else:
            print('Error: mode %s not implemented' % self.mode)
            exit(0)
        out = self.out_layer(concatenated_rep)

        if self.output_feature:
            self.model = Model(inputs=inputs, outputs=[out, concatenated_rep])
        else:
            self.model = Model(inputs=inputs, outputs=out)

        layers = self.model.layers
        layer_dict = dict([(layer.name, layer) for layer in layers])
        self.model.summary()
        for layer_name in self.pretrained_weight_dirs:
            if layer_name in layer_dict:
                layer_dict[layer_name].set_weights([np.loadtxt(weight_dir) if type(weight_dir) == str else weight_dir
                                                    for weight_dir in self.pretrained_weight_dirs[layer_name]])
                print('weight of layer %s successfully loaded.' % layer_name)


if __name__ == '__main__':
    options = ['word']
    nn = NN_architecture(options,
                         word_vocab_size=40000, word_max_len=50,
                         char_vocab_size=1200, char_max_len=150,
                         pretrained_weight_dirs=None, weight_in_keras=None,
                         prefix='shabi', input_format="elmo", elmo_option="add_average")

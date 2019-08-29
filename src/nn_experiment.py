import numpy as np
import subprocess
import os
from data_loader import Data_loader
import time
from sklearn import metrics
from sklearn.metrics import f1_score, precision_recall_fscore_support
from generator_util import create_clf_data, extract_y_from_tweet_dicts


# number of classes we are performing classification task
# currently 3
nb_classes = 3

# all the labeled ids
labeled_tids = np.loadtxt('../data/labeled_tids.np', dtype='int')

# extracting the dimension for each input name
# also assert that all the input from the same input name has the same dimension
# and that for each key input_name2id2np contains mapping from every labeled tweet
def extract_dim_input_name2id2np(input_name2id2np):
    dim_map = {}
    for input_name in input_name2id2np:

        # other input_name
        id2np = input_name2id2np[input_name]
        dim = None
        for id in id2np:
            if dim is None:
                dim = id2np[id].shape[0]
            # asserting the dimension for the same key across all tweets is the same
            assert (id2np[id].shape[0] == dim)
        dim_map[input_name] = dim

        # asserting that labeled tweets has a corresponding input
        for tid in labeled_tids:
            assert (tid in id2np)
    return dim_map


# make remaining predictions: None as 2 ("other" class)
def make_remaining_predictions(arr):
    for idx, val in enumerate(arr):
        if val is None:
            arr[idx] = 2

"""
# OBSOLETE
# for supervised learning without any pre-training
# we need to adjust the vocabulary size for all the content input
# mapping every word that occurs less than twice in the training set to 1
def adapt_vocab(X_train, X_list):
    threshold = 2

    for key in X_train:
        if key in [option + '_content_input' for option in ['char', 'word']]:
            # count the number of occurence for each word
            wc = {}
            for xs in X_train[key]:
                for x in xs:
                    if wc.get(x) is None:
                        wc[x] = 0
                    wc[x] += 1

            # define a filter here
            def f(x):
                return x if (wc.get(x) is not None and wc[x] >= threshold) else 1

            # applying the filter to all x
            X_train[key] = np.array([[f(x) for x in xs] for xs in X_train[key]])
            for X in X_list:
                X[key] = np.array([[f(x) for x in xs] for xs in X[key]])
    """


# an experiment class that runs cross validation
class Experiment:
    def __init__(self, mode, experiment_dir, input_name2id2np=None, adapt_train_vocab=False,
                 epochs=100, patience=4, min_epochs=2, lr=0.003, fold=5, lambda_attn=4, input_format='discrete', by_fold=False,
                 use_generator=False, word_dropper=None,
                 random_mask_option=None, random_mask_prob=0.5, use_rationale=False, rationale_with_UNK=1, comment=None,
                 elmo_representation_dir=None, batch_size=32, **kwargs):
        """
        an experiment class that runs cross validation
        designed to enable easy experiments with combinations of:
        1) context representation:
            handled by input_name2id2np
        2) pre-training methods:
            handled by pretrained_weight_dir in the kwargs argument
            None if there is no pretraining weight available
        Parameters
        ----------
        mode: standard, lstm_attn, cnn_attn
        input_name2id2np:
        experiment_dir: the directory that the experiment weights and results will be saved
        adapt_train_vocab: under supervised training without pretraining,
                            some vocab will not be seen (twice) in the training set.
                            if set to True, then vocab occuring less than twice will be removed.
        comments: the comments that will be written to the README
        epochs: number of epochs of training during cross validation
        patience: number of epochs allowable for not having any improvement on the validation set
        min_epochs: minimum number of epochs of training
        lr: learning rate for the model training
        lambda_attn: the weight of the KL divergence loss for attention, only applicable if the model uses attention
        by_fold: train/test on a single fold or several folds
        fold: if by_fold is true, fold [0-4] means train/test on that single fold. if by_fold is false, fold[1-5] means
        train/test on the first fold number of folds.
        kwargs: arguments that will be passed to initializing the neural network model (shown below)
        """
        # creating the experiment dir
        # automatically generate a README
        assert mode in ['standard', 'lstm_attention', 'cnn_attention']

        if experiment_dir[:-1] != '/':
            experiment_dir += '/'
        experiment_dir = '../experiments/' + experiment_dir
        self.experiment_dir, self.kwargs = experiment_dir, kwargs
        if os.path.exists(self.experiment_dir) is False:
            subprocess.call(['mkdir', self.experiment_dir])

        self.by_fold = by_fold
        if self.by_fold:
            self.experiment_dir += 'fold_%d/' % fold
        subprocess.call(['rm', '-rf', self.experiment_dir])
        subprocess.call(['mkdir', self.experiment_dir])
        subprocess.call(['cp', '-r', '../src', self.experiment_dir + 'src'])

        # parameters for training
        self.mode = mode
        self.adapt_train_vocab = adapt_train_vocab
        if input_name2id2np is None:
            input_name2id2np = {}
        self.input_name2id2np = input_name2id2np
        self.fold = fold
        self.dl = Data_loader(option='both', labeled_only=True, **kwargs)
        self.epochs, self.patience, self.min_epochs, self.lr = epochs, patience, min_epochs, lr
        self.input_format = input_format
        if input_name2id2np is not {}:
            self.kwargs['input_dim_map'] = extract_dim_input_name2id2np(self.input_name2id2np)
        self.kwargs['input_format'] = input_format
        self.input_format = input_format
        if self.mode != "standard":
            self.lambda_attn = lambda_attn
        self.random_mask_option, self.random_mask_prob = random_mask_option, random_mask_prob
        if self.mode != 'standard':
            self.use_rationale = use_rationale
            self.rationale_with_UNK = rationale_with_UNK
        if self.input_format != 'discrete':
            self.elmo_representation_dir = elmo_representation_dir
        self.batch_size = batch_size
        self.use_generator = use_generator
        self.word_dropper = word_dropper

        #record experiment parameters
        experiment_parameters = {'mode': mode, 'input context features': self.input_name2id2np != {}, 'epochs': epochs,
                                 'patience': patience, 'min_epochs': min_epochs, 'lr': lr, 'lambda_attn': lambda_attn,
                                 'input_format': input_format, 'random_mask_option': random_mask_option,
                                 'use generator': use_generator, 'word dropout': str(self.word_dropper),
                                 'random_mask_prob': random_mask_prob, 'use rationale': use_rationale}
        with open(self.experiment_dir + 'README', 'a') as readme:
            readme.write(str(experiment_parameters) + '\n')
        if comment != None:
            with open(self.experiment_dir + 'README', 'a') as readme:
                readme.write(comment + '\n')

    # fitting a binary classification model given the data
    def fit_model(self, tr, X_val, _y_val_, weight_dir):
        assert len(_y_val_.shape) == 1
        assert self.random_mask_option in {'individual', 'sentence', 'sentence_with_prob', None}
        if not self.use_generator:
            X_train, _y_train_ = tr
        else:
            tr_generator, num_steps = tr

        from model_def import NN_architecture
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        if self.mode == 'standard':
            # initialize a model
            architecture = NN_architecture(**self.kwargs)
            self.model = architecture.model
            with open(self.experiment_dir + 'README', 'a') as readme:
                for property_description in architecture.property:
                    readme.write(property_description + '\n')

            self.model.compile(optimizer='adam', loss='binary_crossentropy')

            # call backs
            es = EarlyStopping(patience=self.patience, monitor='val_loss', verbose=1)
            mc = ModelCheckpoint(weight_dir, save_best_only=True, save_weights_only=True)
            callbacks = [es, mc]
            if not self.use_generator:
                # fit for at least one epoch
                self.model.fit(x=X_train, y=_y_train_, validation_data=(X_val, _y_val_),
                               batch_size=self.batch_size)
                self.model.fit(x=X_train, y=_y_train_, batch_size=self.batch_size,
                               validation_data=(X_val, _y_val_),
                               callbacks=callbacks, epochs=self.epochs)
            else:
                self.model.fit_generator(tr_generator, steps_per_epoch=num_steps,
                               validation_data=(X_val, _y_val_), epochs=self.min_epochs - 1)
                self.model.fit_generator(tr_generator, steps_per_epoch=num_steps,
                                         validation_data=(X_val, _y_val_),
                                         callbacks=callbacks, epochs=self.epochs)
            # load back the best model
            self.model.load_weights(weight_dir)

        elif self.mode == 'lstm_attention':
            import torch
            from LSTM_attn import Attn_LSTM
            from model import Model
            torch.set_num_threads(1)

            self.model = Attn_LSTM(**self.kwargs)
            with open(self.experiment_dir + 'README', 'a') as readme:
                for property_description in self.model.property:
                    readme.write(property_description + '\n')

            if not self.use_generator:
                self.model_wrapper = Model(self.model, mode='train', train_X=X_train, train_y=_y_train_, dev_X=X_val, dev_y=_y_val_, model_dir=weight_dir,
                                       output_dir=self.experiment_dir + 'README', num_epochs=self.epochs,
                                       patience=self.patience, min_epochs=self.min_epochs, lr=self.lr, lambda_attn=self.lambda_attn,
                                       use_rationale=self.use_rationale, rationale_with_UNK=self.rationale_with_UNK)
            else:
                self.model_wrapper = Model(self.model, mode='train', tr_generator=tr_generator, num_steps=num_steps, dev_X=X_val,
                                           dev_y=_y_val_, model_dir=weight_dir,
                                           output_dir=self.experiment_dir + 'README', num_epochs=self.epochs,
                                           patience=self.patience, min_epochs=self.min_epochs, lr=self.lr,
                                           lambda_attn=self.lambda_attn,
                                           use_rationale=self.use_rationale, rationale_with_UNK=self.rationale_with_UNK)
            self.model_wrapper.train()

            # load back the best model
            self.model.load_state_dict(torch.load(weight_dir))

        elif self.mode == 'cnn_attention':
            import torch
            from CNN_attn import Attn_CNN
            from model import Model
            torch.set_num_threads(1)

            self.model = Attn_CNN(**self.kwargs)
            self.model_wrapper = Model(self.model, X_train, _y_train_, X_val, _y_val_, model_dir=weight_dir,
                                       output_dir=self.experiment_dir + 'README', num_epochs=self.epochs,
                                       patience=self.patience, min_epochs=self.min_epochs, lr=self.lr,
                                       use_rationale=self.use_rationale, lambda_attn=self.lambda_attn)
            self.model_wrapper.train()

            # load back the best model
            self.model.load_state_dict(torch.load(weight_dir))


    # running train, val, test experiment on a fold/split of data
    def experiment_with(self, data, fold_idx=-1):

        #y_train, y_val are one-hot represented labels of dimension 3, y_test is one-dimensional 0,1,2

        # initializing model, train and predict
        if self.mode == 'standard':
            import keras.backend as K
            K.clear_session()

        self.kwargs['input_dim_map'] = extract_dim_input_name2id2np(self.input_name2id2np)

        # initialize the predictions
        num_train, num_val, num_test = [len(tweet_dicts) for tweet_dicts in data]
        y_val, y_test = [extract_y_from_tweet_dicts(data[i]) for i in [1, 2]]
        y_pred_val, y_pred_test = [None] * num_val, [None] * num_test

        # all the results will be returned in a dictionary
        result_dict = {}

        for class_idx in range(2):
            # create layer name that has prefix
            # since for each fold we train model for aggression and loss models separately
            self.kwargs['prefix'] = 'aggression' if class_idx == 0 else 'loss'

            # time the training
            start = int(round(time.time()))  # seconds
            weight_dir = self.experiment_dir + str(fold_idx) + '_' + str(class_idx) + '.weight'

            (tr, (X_val, _y_val_), X_test) = self.prepare_for_clf(data, class_idx)
            if not self.use_generator:
                # create the label for this binary classification task
                X_train, _y_train_ = tr
            self.fit_model(tr, X_val, _y_val_, weight_dir)

            if self.mode == 'standard':
                _y_pred_val_score, _y_pred_test_score = (self.model.predict(X_val).flatten(),
                                                     self.model.predict(X_test).flatten())
            else:
                _y_pred_val_score, _y_pred_test_score = (self.model_wrapper.predict(X_val),
                                                         self.model_wrapper.predict(X_test))

            # save the prediction scores
            prefix = 'class_%d_scores_' % class_idx
            result_dict[prefix + 'val'] = _y_pred_val_score
            result_dict[prefix + 'test'] = _y_pred_test_score

            # calculate the area under curve
            # and save to the directory
            one_hot_y_test = [1 if b else 0 for b in (y_test == class_idx)]
            fpr, tpr, thresholds = metrics.roc_curve(one_hot_y_test, _y_pred_test_score)
            auc_roc = metrics.auc(fpr, tpr)
            with open(self.experiment_dir + 'README', 'a') as readme:
                readme.write('Fold %d class %d auc_roc: %.4f\n' % (fold_idx, class_idx, auc_roc))

            # threshold tuning
            best_t, best_f_val = 0, -1
            for t in np.arange(0.01, 1, 0.01):
                y_val_pred_ = [0] * num_val
                for idx in range(num_val):
                    if y_pred_val[idx] is None and _y_pred_val_score[idx] >= t:
                        y_val_pred_[idx] = 1
                f = f1_score(_y_val_, y_val_pred_)
                if f > best_f_val:
                    best_f_val = f
                    best_t = t
                # a temp variable that we do not want its value
                # to be accidentally accessed by outside code
                y_val_pred_ = None

            with open(self.experiment_dir + 'README', 'a') as readme:
                readme.write('fold %d class %d threshold best_t: %.2f\n' % (fold_idx, class_idx, best_t))

            # predictions made only when predictions not made by the previous model
            # and larger than the best threshold
            # true for both val_pred and test_pred
            for idx in range(num_val):
                if y_pred_val[idx] is None and _y_pred_val_score[idx] >= best_t:
                    y_pred_val[idx] = class_idx

            for idx in range(num_test):
                if y_pred_test[idx] is None and _y_pred_test_score[idx] >= best_t:
                    y_pred_test[idx] = class_idx

            end = int(round(time.time()))  # seconds

            # write how many time it takes for a run into the readme
            duration = (end - start) // 60  # minutes
            with open(self.experiment_dir + 'README', 'a') as readme:
                readme.write('fold %d class %d takes %d minutes\n'
                             % (fold_idx, class_idx, duration))

        # predict the rest as the "Other" class
        make_remaining_predictions(y_pred_test)
        make_remaining_predictions(y_pred_val)

        # put all the predictions and ground truth in the return dictionary
        result_dict['pred_val'], result_dict['pred_test'], result_dict['truth_val'], result_dict['truth_test'] = \
            y_pred_val, y_pred_test, y_val, y_test

        macro_fscore = f1_score(y_test, y_pred_test, average='macro')
        with open(self.experiment_dir + 'README', 'a') as readme:
            readme.write('Fold %d f score: %.3f\n' % (fold_idx, macro_fscore))

        # save the results of the experiment on this fold
        for key, result_arr in result_dict.items():
            np.savetxt(self.experiment_dir + 'fold_%d_%s.np' % (fold_idx, key), result_arr)

        return result_dict


    def prepare_for_clf(self, data, class_idx):
        #For Keras CNN if we want to use ELMo representation, all elmo representation must be padded to exact shape
        #Setting the pad_elmo flag to be True in create_clf_data method will generate data that pad all elmo representation to max length
        #Padding the elmo representation is performed by create_data in generator_util.

        if self.mode == 'standard' and self.input_format != 'discrete':
            (tr, val, (X_test, _)) = create_clf_data(self.input_name2id2np, data,
                                input_format=self.input_format, class_idx=class_idx,
                                pad_elmo=True, elmo_dir=self.elmo_representation_dir)
        elif self.mode == 'standard' and self.input_format == 'discrete':
            (tr, val, (X_test, _)) = create_clf_data(self.input_name2id2np, data,
                                input_format=self.input_format, class_idx=class_idx,
                                batch_size=self.batch_size, word_dropper=self.word_dropper)
        elif self.input_format != 'discrete':
            (tr, val, (X_test, _)) = create_clf_data(self.input_name2id2np, data,
                                input_format=self.input_format, class_idx=class_idx,
                                pad_elmo=False, elmo_dir=self.elmo_representation_dir, )
        else: #LSTM Attention, discrete input
            if self.use_generator:
                batch_size = self.batch_size
                word_dropper = self.word_dropper
            else:
                batch_size = None
                word_dropper = None
            (tr, val, (X_test, _)) = create_clf_data(self.input_name2id2np, data,
                                input_format=self.input_format, class_idx=class_idx,
                                batch_size=batch_size, word_dropper=word_dropper)

        """
        # if no pretrained weights, adapting vocabulary so that those who appear in
        # X_train less than twice would not be counted
        if self.adapt_train_vocab:
            adapt_vocab(X_train, (X_val, X_test))
            """

        return (tr, val, X_test)


    # cross validation
    # write all results to the directory
    # see read_results for retrieving the performance
    def cv(self):
        if self.by_fold is False:
            results = []

            for fold_idx in range(self.fold):
                print('cross validation fold %d.' % (fold_idx + 1))
                # retriving cross validataion data
                # fold data contains all the information for train, val and test
                fold_data = self.dl.cv_data(fold_idx)

                # train, val, test data ready, train with the given data
                result_dict = self.experiment_with(fold_data, fold_idx=fold_idx)

                # append the result on this fold to results
                results.append(precision_recall_fscore_support(result_dict['truth_test'],
                                                               result_dict['pred_test']))

            # saving results
            results = np.array(results)
            np.savetxt(self.experiment_dir + 'result_by_fold.np', results.flatten())
            np.savetxt(self.experiment_dir + 'result_averaged.np', np.mean(results, axis=0))
            np.savetxt(self.experiment_dir + 'result_std.np', np.std(results, axis=0))

            avg_macro_f = np.mean(np.mean(results, axis=0)[2])
            with open(self.experiment_dir + 'README', 'a') as readme:
                readme.write('macro F-score: %.4f\n' % avg_macro_f)

        else: # train a single fold
            print('cross validation fold %d.' % (self.fold))
            # retriving cross validataion data
            # fold data contains all the information for train, val and test
            fold_data = self.dl.cv_data(self.fold)

            # train, val, test data ready, train with the given data
            result_dict = self.experiment_with(fold_data, fold_idx=self.fold)
            result = precision_recall_fscore_support(result_dict['truth_test'], result_dict['pred_test'])
            with open(self.experiment_dir + 'README', 'a') as readme:
                readme.write('fold macro F-score: %.4f\n' % np.mean(result[2]))


    # examine whether data generalize across time well
    def examine_time_effect(self, group_by_label):
        fold_data = self.dl.data_by_time(group_by_label=group_by_label)
        result_dict = self.experiment_with(fold_data)
        return result_dict



if __name__ == '__main__':
    """
    input_name2id2np = {}
    options = ['word']
    pretrained_weight_dirs = {'aggression_word_embed': ['../weights_average/word_emb_w2v.np'], \
                              'loss_word_embed': ['../weights_average/word_emb_w2v.np']}
    experiment = Experiment(mode='standard', experiment_dir="test",
                            pretrained_weight_dirs=pretrained_weight_dirs,
                            options=options,
                            epochs=100, patience=4, fold=5, input_format='elmo', elmo_option='merge_average',
                            by_fold=False, input_dim_map=input_name2id2np,
                            )
    experiment.cv()
    """

    input_name2id2np = {}
    experiment = Experiment(mode="lstm_attention",
                            experiment_dir='test',
                            input_name2id2np=input_name2id2np,
                            word_embedding_matrix=np.loadtxt('../weights_average/word_emb_w2v_256.np'),
                            use_attn=True,  # whether the model uses attention
                            context_features_before=True,  # whether the model uses context features before
                            context_features_last=True,  # whether the model concatenates context features last
                            epochs=10, patience=4, fold=5, min_epochs=2, by_fold=False, lr=0.03,
                            dropout_embedding=0.5,
                            dropout_lstm=0.5,
                            input_format="both", elmo_option="add_average"
                            )
    experiment.cv()

from copy import deepcopy
import numpy as np

class Word_Dropper:

    def __init__(self, dropout_prob, mode):
        assert mode in (None, 'change_y', 'keep_r')
        self.dropout_prob = dropout_prob
        self.mode = mode

    def dropout(self, generator_output):
        X, y = generator_output
        X = deepcopy(X)
        modified_labels = np.array([1.0 if b else 0.0 for b in y])
        for idx in range(len(X['word_content_input'])):
            x = X['word_content_input'][idx]
            num_words = sum(x != 0)

            # notice that in the mask
            # 1 represents masked position and 0 is non-masked position
            one_hot_mask = [1 if b else 0 for b in np.random.random(num_words) < self.dropout_prob]
            if X['rationale_distr'][idx] is not None and self.mode == 'keep_r':
                # never drop rationales
                r_array = X['rationale_distr'][idx]
                for pos in range(num_words):
                    if r_array[pos] != 0:
                        one_hot_mask[pos] = 0

                # assert none of the rationale dropped
                assert np.sum(one_hot_mask * X['rationale_distr'][idx]) < 0.001

            # find the masked position
            masked_positions = [pos for pos in range(num_words) if one_hot_mask[pos] == 1]

            # mask all masked positions
            X['word_content_input'][idx][masked_positions] = 1

            # decrease confidence of y based on how many rationales are dropped
            if self.mode == 'change_y' and X['rationale_distr'][idx] is not None:
                modified_labels[idx] = 1 - np.sum(one_hot_mask * X['rationale_distr'][idx])
        return X, modified_labels

    def __str__(self):
        return str(self.mode) + '_' + str(self.dropout_prob)

    def __repr__(self):
        return self.__str__()

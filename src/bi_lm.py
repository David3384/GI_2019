import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from torch.nn import Embedding
import collections
import random
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle as pkl
import sys
import gc
import numpy as np

# debug whether all parameters are on gpu (if any)
DEBUG_DEVICE = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

class BiLM:

    def __init__(self,
                 experiment_path,
                 max_len, vocab_size,
                 wembed_dim, hidden_dim, lstm_depth):
        # load the initialization parameters
        self.model_dir = experiment_path + '.weight'
        self.param_dir = experiment_path + '.param'
        self.memory_log = experiment_path + '.memory'
        self.dev_loss_log = experiment_path + '.dev_loss'
        self.max_len, self.vocab_size = max_len, vocab_size
        self.wembed_dim, self.hidden_dim, self.lstm_depth = wembed_dim, hidden_dim, lstm_depth
        # write the parameter to the log file
        # or check whether they are the same if the param file is already there
        self.check_or_write_params()
        
        self.model = MyELMo(vocab_size, wembed_dim, hidden_dim, lstm_depth)
        
        # load model weight if it already exists
        if os.path.exists(self.model_dir):
            print('model load successful')
            self.model = torch.load(self.model_dir, map_location='cpu')
        print('loaded to device')
        self.model.to(DEVICE)

        # debug to maker sure that all parameters are on gpu (if any)
        if DEBUG_DEVICE:
            for param in self.model.parameters():
                print(param.device)
        self.dg = Data_generator(vocab_size, max_len)

    def check_or_write_params(self):
        param_dict = {
            'max_len': self.max_len,
            'vocab_size': self.vocab_size,
            'wembed_dim': self.wembed_dim,
            'hidden_dim': self.hidden_dim,
            'depth': self.lstm_depth
        }
        if os.path.exists(self.param_dir):
            saved_params = pkl.load(open(self.param_dir, 'rb'))
            assert param_dict == saved_params
        else:
            pkl.dump(param_dict, open(self.param_dir, 'wb'))

    def train(self, train_sentences, dev_sentences, epochs=5, batch_size=64, print_every=1000):
        self.batch_size, self.print_every = batch_size, print_every
        self._initialize_optimizer()
        self.dev_sentences = dev_sentences
        self.dg.load_train_data(train_sentences)
        self.num_batches = len(train_sentences) // self.batch_size
        assert self.num_batches >= 1
        
        prev_loss = float('inf')
        dev_loss_history = []

        # train for <epochs> epochs
        for epoch_idx in range(epochs):
            
            print('Training epoch %d: ' % epoch_idx)
            self._train_one_epoch()
            dev_loss = self._eval_loss_on_dev()
            dev_loss_history.append(dev_loss)
            with open(self.dev_loss_log, 'a') as out_file:
                out_file.write('Dev loss = %.3f\n' % dev_loss)
            print('Dev loss = %.3f' % dev_loss)
            # save the model if it has smaller dev loss
            if dev_loss < prev_loss:
                prev_loss = dev_loss
                torch.save(self.model, self.model_dir)
                print('model saved')

    # create represenation for the sentences
    def create_rep(self, sentences):
        self.model.eval()
        data = self.dg.transform_sentences(sentences)
        results = self.model(data)
        return results['rep']

    def evaluate_prob(self, sentences):
        self.model.eval()
        data = self.dg.transform_sentences(sentences)
        results = self.model(data)
        return results['fwd_pred'], results['bw_pred']

    # initialize the optimizer
    def _initialize_optimizer(self):
        self.model.train()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(params, lr=0.01, weight_decay=0.001)

    # train one epoch on the training set
    def _train_one_epoch(self):
        self.model.train()
        print('Finishing batches ')
        for i in range(self.num_batches):
            torch.cuda.empty_cache()
            with open(self.memory_log, 'a') as out:
                memory = torch.cuda.memory_allocated()
                out.write('%d\n' % memory)
            if i % self.print_every == 0 and i != 0:
                print('%d ' % i, end='')
                sys.stdout.flush()
            data = self.dg.generate_batch(batch_size=self.batch_size)
            results = self.model(data)
            
            loss = torch.nn.NLLLoss(ignore_index=0, reduction='sum')(results['fwd_pred'].view((-1, self.vocab_size + 1)),
                                                    data['fwd_truth'].view((-1,)))
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()
            
            loss = torch.nn.NLLLoss(ignore_index=0, reduction='sum')(results['bw_pred'].view((-1, self.vocab_size + 1)), data['bw_truth'].view((-1,)))
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()
        print()

    # evaluate loss on the dev set
    def _eval_loss_on_dev(self):
        self.model.eval()
        total_loss = 0
        num_batches = len(self.dev_sentences) // self.batch_size
        # compute loss in batches
        for idx in range(num_batches):
            data = self.dg.transform_sentences(self.dev_sentences[self.batch_size * idx: self.batch_size * (idx + 1)])
            total_loss += self.compute_loss_on_data(data)
        
        # compute loss for data left
        left = len(self.dev_sentences) % self.batch_size
        if left != 0:
            data = self.dg.transform_sentences(self.dev_sentences[-left:])
            total_loss += self.compute_loss_on_data(data)
        return total_loss / np.sum([len(s) for s in self.dev_sentences])

    def compute_loss_on_data(self, data):
        results = self.model(data)
        loss = 0
        fwd_loss = torch.nn.NLLLoss(ignore_index=0, reduction='sum')(results['fwd_pred'].view((-1, self.vocab_size + 1)), data['fwd_truth'].view((-1,)))
        loss += fwd_loss.item()
        bw_loss = torch.nn.NLLLoss(ignore_index=0, reduction='sum')(results['bw_pred'].view((-1, self.vocab_size + 1)), data['bw_truth'].view((-1,)))
        loss += bw_loss.item()
        return loss


def pad_sentence(sentence, length):
    return (sentence + [0] * length)[:length]

def get_mask(sentence):
    return [1 if w != 0 else 0 for w in sentence]

def create_key2batch(batch_dicts):
    result = collections.defaultdict(list)
    
    # append the corresponding key vector to the final result
    for d in batch_dicts:
        for key in d:
            result[key].append(d[key])

    # cast to pytorch input format
    for key in result:
        if 'mask' not in key:
            result[key] = torch.from_numpy(np.array(result[key])).type(torch.LongTensor).to(DEVICE)
    return result


class Data_generator:

    # vocab size does not include <start> or <end> token
    def __init__(self, vocab_size, max_len):
        self.vocab_size, self.max_len = vocab_size, max_len

    def transform_sentences(self, unpadded_sentences):
        max_len = min(np.max([len(s) for s in unpadded_sentences]), self.max_len)
        batch_dicts = [self._transform_sentence(unpadded_sentence, max_len)
                       for unpadded_sentence in unpadded_sentences]
        return create_key2batch(batch_dicts)

    def _transform_sentence(self, unpadded_sentence, max_len):
        # trim the vocab size
        cp_s = [w if w < self.vocab_size else 1 for w in unpadded_sentence]
        
        # pad the sentences and extract mask
        forward_sentence = pad_sentence([self.vocab_size] + cp_s, max_len)
        forward_train_truth = pad_sentence(cp_s + [self.vocab_size], max_len)
        forward_mask = get_mask(forward_sentence)

        backward_sentence = pad_sentence([self.vocab_size] + cp_s[::-1], max_len)
        backward_train_truth = pad_sentence(cp_s[::-1] + [self.vocab_size], max_len)
        backward_mask = get_mask(backward_sentence)

        return {
            'fwd_s': forward_sentence,
            'fwd_truth': forward_train_truth,
            'fwd_mask': forward_mask,
            'bw_s': backward_sentence,
            'bw_truth': backward_train_truth,
            'bw_mask': backward_mask
        }

    # load the training data, shuffle, and maintain a pointer to where the next block starts
    def load_train_data(self, sentences):
        self.train_sentences = sentences
        random.shuffle(self.train_sentences)
        self.cur_idx = 0

    def generate_batch(self, batch_size=64):
        assert batch_size <= len(self.train_sentences)
        # sequentially read each block
        if self.cur_idx + batch_size <= len(self.train_sentences):
            data = self.transform_sentences(self.train_sentences[self.cur_idx: self.cur_idx + batch_size])
            self.cur_idx = (self.cur_idx + batch_size) % len(self.train_sentences)
            return data
        # if "overflow", read the rest (block1), shuffle an dthen read block2
        # concatenate and transform
        else:
            next_idx = (self.cur_idx + batch_size) % len(self.train_sentences)
            # read the rest of the block
            sentence_block1 = self.train_sentences[self.cur_idx:]
            # shuffle for the next epoch
            random.shuffle(self.train_sentences)
            sentence_block2 = self.train_sentences[:next_idx]
            
            # concat two blocks
            sentences = sentence_block1 + sentence_block2
            self.cur_idx = next_idx
            data = self.transform_sentences(sentences)
            return data


# stacked lstm
# forward returns the representation of each layer
class Stacked_lstm(nn.Module):

    def __init__(self, input_dim, hidden_dim, lstm_depth):
        super(Stacked_lstm, self).__init__()
        self.lstms = nn.ModuleList()
        self.lstms.append(nn.LSTM(input_dim, hidden_dim, num_layers=1,
                                  bidirectional=False))
        assert lstm_depth >= 1
        for depth in range(lstm_depth - 1):
            self.lstms.append(nn.LSTM(hidden_dim, hidden_dim,
                                      num_layers=1, bidirectional=False))

    def forward(self, seqs, masks):
        # sort all the sequences by length
        lengths = np.array([np.sum(mask) for mask in masks])
        perm = np.argsort(-lengths)
        perm_inv = np.argsort(perm)
        
        # pack and forward
        seq_perm, lengths_perm = seqs[perm], lengths[perm]

        # compute representations
        representations = []
        packed_seq = pack_padded_sequence(seq_perm, lengths_perm, batch_first=True)
        last_rep = packed_seq
        
        # passing through different layers
        for idx, lstm in enumerate(self.lstms):
            packed_output, _ = lstm(last_rep)
            last_rep = packed_output
            representations.append(packed_output)

        for layer_idx in range(len(representations)):
            representations[layer_idx], _ = pad_packed_sequence(representations[layer_idx],
                                                                batch_first=True)
            representations[layer_idx] = representations[layer_idx][perm_inv]
        return representations


class Uni_lm(nn.Module):

    def __init__(self, vocab_size, wembed_dim, hidden_dim, lstm_depth):
        super(Uni_lm, self).__init__()
        # word emebdding
        # the one more token is "<start>" for forward
        self.wemb = Embedding(vocab_size + 1, wembed_dim)
        
        # the forward stacked lstm
        self.slstm = Stacked_lstm(wembed_dim, hidden_dim, lstm_depth)
    
        # last fully connected layer
        # the one more token is "<end>" for forward
        self.fc = nn.Linear(hidden_dim, vocab_size + 1)
        self.lsm = nn.LogSoftmax(2)
    
    # masks is a list of list of 0/1
    # sentences is a list of list of discrete indexes (tokens)
    # with boundary already padded in the front
    def forward(self, sentences, masks):
        embedded = self.wemb(sentences)
        all_rep = self.slstm(embedded, masks)
        final_rep = all_rep[-1]
        log_prob = self.lsm(self.fc(final_rep))
        for tweet in log_prob:
            x = tweet.detach().cpu().numpy()
        batch_size, seq_length = sentences.shape
        batch_size, seq_length, _ = log_prob.shape
        return {'log_prob': log_prob, 'all_rep': all_rep}

class MyELMo(nn.Module):

    # vocab size not including <start>, <end>
    # vocab size already includes pad
    def __init__(self, vocab_size, wembed_dim, hidden_dim, lstm_depth):
        super(MyELMo, self).__init__()
        self.lms = nn.ModuleDict()
        self.lms['fwd'] = Uni_lm(vocab_size, wembed_dim, hidden_dim, lstm_depth)
        self.lms['bw'] = Uni_lm(vocab_size, wembed_dim, hidden_dim, lstm_depth)

    # forward sentences to get representations
    # and log prob
    def forward(self, key2batch):
        results = {}
        # collect results from both forward and backward Language Model
        for key in self.lms:
            model_forward = self.lms[key](key2batch[key + '_s'], key2batch[key + '_mask'])
            results[key + '_pred'], results[key + '_rep'] = model_forward['log_prob'], model_forward['all_rep']

        # concatenate each layer of forward/backward representation
        results['rep'] = []
        for fwd_rep, bw_rep in zip(results['fwd_rep'], results['bw_rep']):
            rep = torch.cat((fwd_rep, bw_rep), -1)
            results['rep'].append(rep)
        
        # delete individual representation for easiness
        for key in self.lms:
            del results[key + '_rep']
        # import pdb
        # pdb.set_trace()
        return results

if __name__ == '__main__':
    # intializing test case
    masks = np.array([[1,1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
    num_seqs, seq_length = masks.shape
    input_dim, hidden_dim, depth = 50, 64, 3
    vocab_size, wembed_dim, max_len = 100, 20, 7
    seq = np.random.normal(size=(num_seqs, seq_length, input_dim))
    for seq_idx, seq_mask in enumerate(masks):
        for element_idx, mask in enumerate(seq_mask):
            if masks[seq_idx][element_idx] == 0:
                seq[seq_idx][element_idx] = np.zeros(shape=(input_dim, ))
    seq = torch.from_numpy(seq).type(torch.FloatTensor)

    """
    print('debugging stack lstm')
    # defining a model here
    model = Stacked_lstm(input_dim, hidden_dim, depth)
    output = model(seq, masks)
    print(output[0])
    """

    """
    print('debugging uni direction lm')
    seqs = torch.LongTensor(np.random.randint(vocab_size, size=(num_seqs, seq_length)) * masks)
    print(seqs)
    u_lm = Uni_lm(vocab_size, wembed_dim, hidden_dim, depth)
    result = u_lm(seqs, masks)
    print(result)
    """

    # seqs = [[1,3,4,101, 100, 2, 3], [11,12,13], [21], [22,24,27,40]]
    seqs = np.random.randint(99, size=(10000, max_len)).tolist()
    
    dg = Data_generator(vocab_size, max_len)
    data = dg.transform_sentences(seqs)

    """
    print('debugging data generator')
    for key in data:
        print(key)
        print(data[key])

    dg.load_train_data(seqs)
    for _ in range(10):
        data = dg.generate_batch(batch_size=3)
    """

    print('debugging MyELMo')
    el = MyELMo(vocab_size, wembed_dim, hidden_dim, depth)
    el.to(DEVICE)
    results = el(data)

    print(results['fwd_pred'].shape)
    print(results['fwd_pred'].device)
    print(results['fwd_pred'].view((-1, vocab_size + 1)).shape)
    print(data['fwd_truth'].view((-1,)).shape)
    l = torch.nn.NLLLoss(ignore_index=0)(results['fwd_pred'].view((-1, vocab_size + 1)), data['fwd_truth'].view((-1,)))
    print(l)

    print('Debugging bi-lm wrapper')
    experiment_path = './bilmtest'
    bilm = BiLM(experiment_path, max_len, vocab_size, wembed_dim, hidden_dim, depth)
    bilm.train(seqs, seqs, epochs=1000)

import os
import torch
import csv


class DataHelper(object):
    def __init__(self, dataset, mode, vocab=None):
        allowed_data = ['r8', 'r52', 'mr', 'oh', 'SST1', 'SST2', '20ng', 'ag_news', 'dblp', 'aclImdb', 'TREC', 'WebKB', 'wiki', 'twitter']

        if dataset not in allowed_data:
            raise ValueError('currently allowed data: %s' % ','.join(allowed_data))
        else:
            self.dataset = dataset

        self.mode = mode
        self.base = '/content/data/' + self.dataset
        self.current_set = os.path.join(self.base, '%s-%s.txt' % (self.dataset, self.mode))

        with open(os.path.join(self.base, 'label.txt')) as f:
            labels = f.read()
        self.labels_str = labels.split('\n')

        content, label = self.get_content()

        self.label = self.label_to_onehot(label)
        if vocab is None:
            self.vocab = []

            try:
                self.get_vocab()
            except FileNotFoundError:
                self.build_vocab(content, min_count=5)
        else:
            self.vocab = vocab

        self.d = dict(zip(self.vocab, range(len(self.vocab))))

        self.content = [list(map(lambda x: self.word2id(x), doc.split(' '))) for doc in content]

    def label_to_onehot(self, label_str):

        return [self.labels_str.index(l) for l in label_str]

    def get_content(self):
        with open(self.current_set) as f:
            all = f.read()
            content = [line.split('\t') for line in all.split('\n')]
       
        label, content = zip(*content)

        return content, label

    def word2id(self, word):

        try:
            result = self.d[word]
        except KeyError:
            result = self.d['UNK']

        return result

    def get_vocab(self):
        with open(os.path.join(self.base, self.dataset+'-vocab.txt')) as f:
            vocab = f.read()
            self.vocab = vocab.split('\n')

    def batch_iter(self, batch_size, num_epoch):
        for i in range(num_epoch):
            num_per_epoch = int(len(self.content) / batch_size)
            for batch_id in range(num_per_epoch):
                start = batch_id * batch_size
                end = min((batch_id + 1) * batch_size, len(self.content))

                content = self.content[start:end]
                label = self.label[start:end]

                yield content, torch.tensor(label).cuda(), i

  

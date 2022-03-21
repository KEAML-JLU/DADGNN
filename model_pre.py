import dgl
import torch
import torch.nn.functional as F
import numpy as np
import gensim
from attention_diffusion import GATNet
from dgl.nn.pytorch.glob import WeightAndSum


class Model(torch.nn.Module):
    def __init__(self,
                
                 
                 num_hidden,
                 num_layers,
                 num_heads,
                 k,
                 alpha,
                 vocab,
                 n_gram,
                 drop_out,
                 class_num,
                 num_feats,
                 max_length=350,
                 cuda=True,
                 ):
        super(Model, self).__init__()

        self.is_cuda = cuda
        self.vocab = vocab

        self.node_hidden = torch.nn.Embedding(len(vocab), num_feats)
        self.node_hidden.weight.data.copy_(torch.tensor(self.load_word2vec('/content/glove.6B.300d.txt')))
        self.node_hidden.weight.requires_grad = True

        self.len_vocab = len(vocab)
        self.ngram = n_gram
        self.max_length = max_length
        
        self.gatnet = GATNet(class_num, class_num, class_num, num_layers, k, alpha, num_heads, merge='mean')
        self.dropout = torch.nn.Dropout(p=drop_out)
        self.activation = torch.nn.ReLU()
        
        #self.attn_fc = torch.nn.Linear(2 * hidden_size_node, 1, bias=False)
        
        self.linear1 = torch.nn.Linear(num_feats, num_hidden, bias=True)
        self.bn1 = torch.nn.BatchNorm1d(num_hidden)
        self.linear2 = torch.nn.Linear(num_hidden, class_num, bias=True)
        self.bn2 = torch.nn.BatchNorm1d(class_num)        
        self.weight_and_sum = WeightAndSum(class_num)
        #self.bn = torch.nn.BatchNorm1d(hidden_size_node)
        #self.ln = torch.nn.LayerNorm(hidden_size_node)
        #self.reset()
    
    def reset(self):
      gain = torch.nn.init.calculate_gain("relu")
      torch.nn.init.xavier_normal_(self.Linear.weight, gain=gain)
      torch.nn.init.xavier_normal_(self.gate_nn.weight, gain=gain)
      torch.nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def load_word2vec(self, word2vec_file):
        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file)

        embedding_matrix = []

        for word in self.vocab:  
            try:
                embedding_matrix.append(model[word])
            except KeyError:
                embedding_matrix.append(np.random.uniform(-0.1,0.1,300))

        embedding_matrix = np.array(embedding_matrix)

        return embedding_matrix

    def add_seq_edges(self, doc_ids: list, old_to_new: dict):

        edges = []
        old_edge_id = []
        
        for index, src_word_old in enumerate(doc_ids):
            src = old_to_new[src_word_old]
            for i in range(max(0, index - self.ngram), min(index + self.ngram + 1, len(doc_ids))):
                dst_word_old = doc_ids[i]
                dst = old_to_new[dst_word_old]
                edges.append([src, dst])
             
        '''
        for index, src_word_old in enumerate(list(reversed(doc_ids))):

            src = old_to_new[src_word_old]
            for i in range(max(0, index - self.ngram), min(index, len(doc_ids))):
                dst_word_old = doc_ids[i]
                dst = old_to_new[dst_word_old]

                # - first connect the new sub_graph
                edges.append([src, dst])
                # - then get the hidden from parent_graph
                try :
                 old_edge_id.append(self.edges_matrix[(src_word_old, dst_word_old)])
                except KeyError:
                 old_edge_id.append(np.random.randint(0,self.edges_num))
        '''
        return edges

    def seq_to_graph(self, doc_ids: list) -> dgl.DGLGraph():
        
        if len(doc_ids) > self.max_length:
            doc_ids = doc_ids[:self.max_length]
        
        local_vocab = set(doc_ids)

        old_to_new = dict(zip(local_vocab, range(len(local_vocab))))

        if self.is_cuda:
            local_vocab = torch.tensor(list(local_vocab)).cuda()
        else:
            local_vocab = torch.tensor(list(local_vocab))

        sub_graph = dgl.DGLGraph()
        sub_graph.add_nodes(len(local_vocab))
        local_node_hidden = self.node_hidden(local_vocab)

        sub_graph.ndata['k'] = local_node_hidden
        seq_edges = self.add_seq_edges(doc_ids, old_to_new)
        edges = []
        edges.extend(seq_edges)
        srcs, dsts = zip(*edges)
        sub_graph.add_edges(srcs, dsts)
        
        return sub_graph

    def forward(self, doc_ids):

        sub_graphs = [self.seq_to_graph(doc) for doc in doc_ids]
        
        batch_graph = dgl.batch(sub_graphs)
        
        #h1 = dgl.sum_nodes(batch_graph, feat='h')
        batch_f = self.dropout(batch_graph.ndata['k'])
        batch_f = self.activation(self.linear1(batch_f))
        batch_f = self.linear2(self.dropout(batch_f))
        h1 = self.gatnet(batch_graph, batch_f)
        h1 = self.weight_and_sum(batch_graph, h1)
        #h1 = self.set_trans_dec(batch_graph, batch_f)
        #drop1 = self.dropout(h1)
        #drop1 = self.bn(drop1)
        #act1 = self.activation(h1)
        #l = self.Linear(act1)
     
        return h1

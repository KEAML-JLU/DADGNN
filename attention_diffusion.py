import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SingleHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, k, alpha, activation=F.elu, layer_norm=False, batch_norm=False, residual=False, dropout=0):
        super(SingleHeadGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=True)
        self._k = k
        self._alpha = alpha
        self.activation = activation
        self.batch_norm = batch_norm
        self.residual = residual
        self.layer_norm = layer_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(in_dim)
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        TEMP = alpha*(1-alpha)**np.arange(k+1)
        TEMP[-1] = (1-alpha)**k
        self.temp = nn.Parameter(th.tensor(TEMP))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        
        nn.init.zeros_(self.temp)
        for k in range(self._k+1):
            self.temp.data[k] = self._alpha*(1-self._alpha)**k
        self.temp.data[-1] = (1-self._alpha)**self._k

    def edge_attention(self, edges):
        z2 = th.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}
  
    def forward(self, graph, features):
      with graph.local_scope():
        if self.layer_norm:
          h_tide = self.ln(features)
        else:
          h_tide = features
        features_two = features * self.temp[0]
        for k in range(self._k):
          
          graph.ndata['z'] = features
          graph.apply_edges(self.edge_attention)
          e = graph.edata.pop('e')
          graph.edata['w'] = self.dropout(edge_softmax(graph,e))
          
          graph.update_all(fn.u_mul_e('z', 'w', 'm'), fn.sum('m', 'h'))
          features = graph.ndata.pop('h')
          gamma = self.temp[k+1]
          features_two = features_two + gamma * features

        if self.batch_norm:
             features = self.bn(features)
        if self.activation:
             features = self.activation(features)
        if self.residual:
             features = features + self.res_fc(h_tide)
        return features_two


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, k, alpha, num_heads, merge='cat', activation=F.elu,
                 batch_norm=False, residual=False, dropout=0):
        super(GATLayer, self).__init__()
        self.heads = nn.ModuleList()
        self.Wo = nn.Linear(num_heads*out_dim, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()
        layer_norm = False
        
        for i in range(num_heads):
            self.heads.append(SingleHeadGATLayer(in_dim, out_dim, k, alpha, activation, layer_norm, batch_norm, residual, dropout))
        self.merge = merge
        self.reset_parameters()
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.Wo.weight, gain=gain)
        
    def forward(self, g, features):
        head_outs = [attn_head(g, features) for attn_head in self.heads]
        if self.merge == 'cat':
            h_hat = self.Wo(th.cat(head_outs, dim=1))
            
        else:
            h_hat = th.mean(th.stack(head_outs), dim=0)
        return h_hat
            

class GATNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, k, alpha, num_heads, merge='cat',
                 activation=F.elu, batch_norm=False, residual=False, dropout=0.5):
        super(GATNet, self).__init__()
        self.num_layers = num_layers
        self.s = nn.Parameter(th.FloatTensor(num_classes,1))
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(num_feats, num_hidden, k, alpha, num_heads, merge,
                                    activation, batch_norm, residual, dropout))
        for i in range(1, num_layers):
            self.layers.append(GATLayer(num_hidden, num_hidden, k, alpha, num_heads, merge,
                                        activation,batch_norm, residual, dropout))

        self.reset_parameters()
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_uniform_(self.s, gain=gain)

    def forward(self, graph, features):
        h = features
        graph = graph.local_var()
        results = [h]
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
            results.append(h)
        return h

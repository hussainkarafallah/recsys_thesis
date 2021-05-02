import torch as th
import torch.nn as nn
import torch.nn.functional as thF
import dgl.function as gF
from tqdm import tqdm
import numpy as np
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.model.init import xavier_normal_initialization

import commons

class StochasticGCMC(GeneralRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(StochasticGCMC, self).__init__(config, dataset)

        # load dataset info
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.cpu_graph = dataset.graph
        self.cpu_graph.ndata['fts'] = th.eye(self.cpu_graph.num_nodes())
        self.fans = config['fans']
        assert min(self.cpu_graph.in_degrees()) > 0

        # load parameters info
        self.layers_dim = config['layers']
        self.num_layers = len(self.layers_dim)
        self.loss = nn.CrossEntropyLoss()

        # parameters initialization
        self.W1 = nn.Linear(self.cpu_graph.num_nodes() , 512)
        self.W2 = nn.Linear(self.cpu_graph.num_nodes() , 512)
        self.W_glob = nn.Linear(512 , 75)
        self.decoder = DenseBiDecoder(75 , 2)
        self.reset_parameters()

    def reset_parameters(self):
        from torch.nn.init import xavier_uniform_
        xavier_uniform_(self.W1.weight)
        xavier_uniform_(self.W2.weight)
        xavier_uniform_(self.W_glob.weight)
        self.decoder.reset_parameters()

    @staticmethod
    def msg(edges):
        return {
            'm' : edges.src['norm'] * edges.src['x'] * edges.dst['norm']
        }

    def forward_block(self , block , h):
        with block.local_scope():
            block.srcdata['x'] = h
            block.update_all(self.msg , gF.sum('m' , 'y'))
            x = thF.relu(block.dstdata['y'])
            x = thF.relu(self.W_glob(x))
            return x


    def forward(self , batch):
        users, pos, neg, blocks = batch
        split = users.max() + 1
        h = th.cat((self.W1(blocks[0].srcdata['fts'][:split]), self.W2(blocks[0].srcdata['fts'][split:])), dim=0)
        for block in blocks:
            h = self.forward_block(block, h)

        users, pos, neg = h[users], h[pos], h[neg]
        predictions = self.decoder(th.cat((users, users)), th.cat((pos, neg)))
        return predictions

    def calculate_loss(self, batch):
        users, pos, neg, blocks = batch
        predictions = self.forward(batch)
        target = th.zeros(len(users) * 2, dtype=th.long).to(self.device)
        target[:len(users)] = 1

        return self.loss(predictions , target)

    @th.no_grad()
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID] + self.num_users
        user , item = self.check_point[user] , self.check_point[item]          # [batch_size, embedding_size]
        ret = self.decoder(user , item)[:,1]
        return ret.cpu()

    @th.no_grad()
    def inference(self,mode='validation'):
        """
        Offline inference with this module
        """
        print(mode)
        assert mode in ['validation' , 'testing'] , "got mode {}".format(mode)
        from dgl.dataloading import NodeDataLoader , MultiLayerNeighborSampler
        self.eval()
        if mode == 'testing':
            sampler = MultiLayerNeighborSampler([None])
        else:
            sampler = MultiLayerNeighborSampler(self.fans)
        g = self.cpu_graph
        self.check_point = th.zeros(g.number_of_nodes(),75).to(commons.device)
        kwargs = {
            'batch_size' : 64,
            'shuffle' : True,
            'drop_last' : False,
            'num_workers' : 6,
        }
        dataloader = NodeDataLoader(g,th.arange(g.number_of_nodes()), sampler,**kwargs)

        x = self.cpu_graph.ndata['fts'].to(commons.device)
        x = th.cat((self.W1(x[:self.num_users]), self.W2(x[self.num_users:])), dim=0)

        # Within a layer, iterate over nodes in batches
        for input_nodes, output_nodes, blocks in tqdm(dataloader):
            block = blocks[0].to(commons.device)
            h = self.forward_block(block , x[input_nodes])
            self.check_point[output_nodes] = h

        print('Inference Done Successfully')


class DenseBiDecoder(nn.Module):
    def __init__(self,in_units,num_classes,num_basis=2,dropout_rate=0.0):
        super().__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.P = nn.Parameter(th.randn(num_basis, in_units, in_units))
        self.combine_basis = nn.Linear(self._num_basis, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ufeat, ifeat):
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        out = th.einsum('ai,bij,aj->ab', ufeat, self.P, ifeat)
        out = self.combine_basis(out)
        return out
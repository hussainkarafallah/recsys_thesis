from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from utils.decorators import thread_wrapped_func
import torch.multiprocessing as mp
import commons
import torch as th
import torch.nn.functional as thF
import numpy as np
import random , logging , os
from recbole.model.loss import BPRLoss
import logging


class DeepWalk(GeneralRecommender):

    input_type = InputType.PAIRWISE
    __name__ = 'DeepWalk'
    default_params = {
        'num_walks': 100,
        'walk_length': 50,
        'embeddings': 64,
        'window': 5,
        'epochs': 1,
    }
    def __init__(self, config, dataset):

        super(DeepWalk, self).__init__(config, dataset)

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.graph = dataset.graph

        self.num_walks = config['num_walks']
        self.walk_length = config['walk_length']
        self.dimensions = config['embeddings']
        self.window_size = min(self.walk_length , config['window'])
        self.seed = commons.seed
        self.logger = logging.getLogger()
        self.epochs = config['passes']

        walkhash = config['dataset'] + "_n_" + str(self.num_walks) + "_l_" + str(self.walk_length)
        embeddinghash = walkhash + "_w_" + str(self.window_size) + "_d_" + str(self.dimensions) + "_e_" + str(self.epochs)

        self.walks_file = os.path.join(commons.root_dir , "data/walks" , "deepwalk_{}.walks".format(walkhash))
        self.embeddings_file = os.path.join(commons.root_dir , "data/walks" , "deepwalk_{}".format(embeddinghash))
        if not os.path.exists(self.walks_file):
            self.logger.info("No cached walks were found, started generating.")
            self.generate_walks()
        else:
            self.logger.info("Found cached walks on disk.")
        self.dummy_parameter = th.nn.Parameter(th.zeros((2,2) , requires_grad=True))
        self.fit()
        self.init_params()

    def init_params(self):
        self.embeddings = thF.normalize(th.from_numpy(self._embedding), p=2, dim=1)
        self.embeddings = th.nn.Parameter(self.embeddings, requires_grad=False)

    @staticmethod
    @thread_wrapped_func
    def sample(g , nodes , length , queue : mp.Queue):
        from dgl.sampling import random_walk
        ret = random_walk(g , nodes , length=length)[0].numpy().tolist()
        queue.put(ret)
        return None

    @staticmethod
    def simple_filter(x):
        ret = [str(y) for y in x if y != -1]
        return ret

    def generate_walks(self):

        g = self.graph
        assert g.is_homogeneous

        all_nodes = g.nodes().numpy().tolist() * self.num_walks
        random.shuffle(all_nodes)

        queue = mp.JoinableQueue()
        per_worker = len(all_nodes) // commons.workers + 1
        ps = []
        for i in range(commons.workers):
            chunk = all_nodes[i * per_worker : (i+1) * per_worker]
            ps.append(mp.Process(target=self.sample , args=(g , chunk , self.walk_length , queue))),

        for p in ps:
            p.start()

        all_walks = []
        for i in range(commons.workers):
            all_walks.extend(queue.get())

        for p in ps:
            p.terminate()

        with open(self.walks_file , 'w') as f:
            for walk in all_walks:
                walk = self.simple_filter(walk)
                f.write(' '.join(walk))
                f.write('\n')



    def fit(self):
        from gensim.models.word2vec import Word2Vec
        try:
            self._embedding = np.load(self.embeddings_file + ".npy")
            self.logger.info("Embeddings found on disk")

        except Exception:
            self.logger.info("Embeddings not found on disk training language model")
            model = Word2Vec(
                             corpus_file=self.walks_file,
                             hs=1,
                             sg=1,
                             epochs = self.epochs,
                             vector_size=self.dimensions,
                             window=self.window_size,
                             min_count=1,
                             workers=commons.workers,
                             seed=self.seed
                    )

            g = self.graph
            num_of_nodes = g.num_nodes()
            self._embedding = np.array([model.wv[str(n)] for n in range(num_of_nodes)])
            np.save(self.embeddings_file , self._embedding)


    def predict(self, interaction):
        user = interaction[self.USER_ID]
        assert user.device.type == 'cuda'
        item = interaction[self.ITEM_ID] + self.num_users
        user , item = self.embeddings[user] , self.embeddings[item]
        ret = th.mul(user , item).sum(dim = 1).squeeze()
        return ret

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        u_embeddings = self.embeddings[user]
        i_embeddings = self.embeddings[self.num_users :]
        assert i_embeddings.shape[0] == self.num_items
        scores = th.matmul(u_embeddings, i_embeddings.T)
        return scores.view(-1)

    def forward(self , *input , **kwargs):
        pass

class ExtendedDeepWalk(DeepWalk):
    __name__ = 'DeepWalk++'
    default_params = {
        'num_walks': 100,
        'walk_length': 5,
        'embeddings': 128,
        'window': 5,
        'dropout' : 0.2,
        'stopping_step' : 10,
    }

    def __init__(self , config , dataset):
        super(ExtendedDeepWalk, self).__init__(config , dataset)
        self.loss = BPRLoss()
        self.dropout_rate = config['dropout']
        self.dropout = th.nn.Dropout(self.dropout_rate)

    def init_params(self):
        from torch.nn.init import kaiming_uniform_ , zeros_
        self.embeddings = th.nn.Parameter(th.from_numpy(self._embedding) , requires_grad=False)
        self.W = th.nn.Linear(self.dimensions , self.dimensions)
        kaiming_uniform_(self.W.weight)
        zeros_(self.W.bias)

    def forward(self , idxes):
        x = self.embeddings[idxes]
        x = self.dropout(x)
        ret = thF.relu(self.W(x))
        return ret
        #return thF.normalize(ret , p = 2 , dim = 1)

    def calculate_loss(self, interaction):
        users = interaction[self.USER_ID]
        pos = interaction[self.ITEM_ID] + self.num_users
        neg = interaction[self.NEG_ITEM_ID]+ self.num_users

        users , pos , neg = map(self.forward , [users , pos , neg])

        pos_item_score = th.mul(users, pos).sum(dim=1)
        neg_item_score = th.mul(users, neg).sum(dim=1)

        return self.loss(pos_item_score , neg_item_score)

    def predict(self, interaction):
        user = self.forward(interaction[self.USER_ID])
        item = self.forward(interaction[self.ITEM_ID] + self.num_users)
        return th.mul(user , item).sum(dim = 1).cpu()

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        u_embeddings = self.embeddings[user]
        i_embeddings = self.embeddings[self.num_users :]
        assert i_embeddings.shape[0] == self.num_items
        scores = th.matmul(u_embeddings, i_embeddings.T)
        return scores.view(-1)

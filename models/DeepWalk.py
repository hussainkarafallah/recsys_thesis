from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from utils.decorators import thread_wrapped_func
import torch.multiprocessing as mp
import commons
import torch as th
import torch.nn.functional as thF
import numpy as np
import random , logging , os

class DeepWalk(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DeepWalk, self).__init__(config, dataset)

        for log_name, log_obj in logging.Logger.manager.loggerDict.items():
            if log_name != '<module name>':
                log_obj.disabled = True

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.graph = dataset.graph

        self.num_walks = config['num_walks']
        self.walk_length = config['walk_length']
        self.dimensions = config['embeddings']
        self.window_size = config['window']
        self.learning_rate = config['learning_rate']
        self.seed = commons.seed
        self.logger = logging.getLogger(commons.logger_name)


        arghash = hash((self.num_walks , self.walk_length))
        self.walks_file = os.path.join(commons.root_dir , "data/walks" , "deepwalk_{}.walks".format(arghash))
        self.embeddings_file = os.path.join(commons.root_dir , "data/walks" , "deepwalk_{}".format(arghash))
        if not os.path.exists(self.walks_file):
            self.logger.info("No cached walks were found, started generating.")
            self.generate_walks()
        else:
            self.logger.info("Found cached walks on disk.")
        self.dummy_parameter = th.nn.Parameter(th.zeros((2,2) , requires_grad=True))

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
        print(len(all_nodes))
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

        all_walks = [ self.simple_filter(x) for x in all_walks ]
        with open(self.walks_file , 'w') as f:
            for walk in all_walks:
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
                             alpha=self.learning_rate,
                             epochs = 1,
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

        self.embeddings = thF.normalize(th.from_numpy(self._embedding) , p = 2 , dim = 1)

    def predict(self, interaction):
        interaction = interaction.cpu()
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID] + self.num_users
        user , item = self.embeddings[user] , self.embeddings[item]          # [batch_size, embedding_size]
        ret = th.mul(user , item).sum(dim = 1).squeeze()
        return ret.cpu()



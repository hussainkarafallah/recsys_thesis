#%%

from logging import getLogger
from recbole.utils import init_seed , init_logger
from training.static_trainer import StaticTrainer
from recbole.config import Config
from recbole.data import data_preparation , create_dataset
from utils.data import add_graph
import pandas as pd
import commons

#%%

from models.StochasticGCMC import StochasticGCMC

from models.DeepWalk import DeepWalk
config_dict = {
    'num_walks' : 200,
    'walk_length' : 20,
    'embeddings' : 128,
    'window' : 5,
    'learning_rate' : 0.025,
    'min_user_inter_num' : 5,
    'min_item_inter_num' : 5,
    'eval_setting': 'RO_RS,pop100',
    'epochs' : 1
}

config = Config(model = DeepWalk , dataset='ml-100k' , config_dict=config_dict)
init_seed(config['seed'], config['reproducibility'])

# logger initialization
init_logger(config)
logger = getLogger()


logger.info(config)

# dataset filtering
dataset = create_dataset(config)
logger.info(dataset)

train_data, valid_data, test_data = data_preparation(config, dataset)
train_data = add_graph(train_data)
#%%
model = DeepWalk(config, train_data)
logger.info(model)
#%%
trainer = StaticTrainer(config , model)
trainer.fit(train_data , valid_data)

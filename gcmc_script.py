
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import data_preparation , create_dataset
import pandas as pd

#
from models.StochasticGCMC import StochasticGCMC
from utils.data import add_graph

config_dict = {
    'train_batch_size' : 256,
    'layers' : [64] ,
    'fans' : [10],
    'min_user_inter_num' : 5,
    'min_item_inter_num' : 5,
    'eval_setting': 'RO_RS,pop100',
    'learning_rate': 0.001,
}

config = Config(model = StochasticGCMC , dataset='ml-100k' , config_dict=config_dict)
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


model = StochasticGCMC(config, train_data).to(config['device'])
logger.info(model)

#print(min(train_data.inter_matrix().row))
#print(max(train_data.inter_matrix().row))


from training.stochastic_trainer import StochasticTrainer
trainer = StochasticTrainer(config , model)

best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
test_result = trainer.evaluate(test_data , mode='testing')

logger.info('best valid result: {}'.format(best_valid_result))
logger.info('test result: {}'.format(test_result))

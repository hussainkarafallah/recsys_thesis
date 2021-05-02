from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from models.MF import NewModel
from recbole.model.general_recommender import GCMC
from recbole.config import Config
from recbole.data import data_preparation
from utils.data import create_dataset


if __name__ == '__main__':

    config = Config(model=NewModel)
    init_seed(config['seed'], config['reproducibility'])

    print(config.dataset)


    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = NewModel(config, train_data).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))

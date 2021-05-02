from models.DeepWalk import DeepWalk
from recbole.config import Config
from recbole.utils import init_seed , init_logger
from recbole.data import data_preparation , create_dataset
from utils.data import add_graph
import statics
import argparse , traceback , json


parser = argparse.ArgumentParser()
parser.add_argument("--model" , type = str , action = 'store' , help = "model name")
parser.add_argument("--dataset" , type = str , action = 'store' , help = "dataset name")
args , unknown = parser.parse_known_args()


model_name = args.model
dataset_name = args.dataset
model_class = None
if args.model == 'DeepWalk':
    model_class = DeepWalk

def run_trial(hp_config):

    default_config = model_class.default_params
    default_config.update(statics.datasets_params[dataset_name])
    default_config.update(hp_config)

    config = Config(model=model_class, dataset=dataset_name, config_dict=default_config)
    init_seed(config['seed'], config['reproducibility'])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    train_data = add_graph(train_data)

    model = model_class(config, train_data)
    trainer = statics.trainers[args.model](config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    test_result = trainer.evaluate(test_data)
    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


from itertools import product

hp_dict = statics.hyperparameters[args.model]
keys = hp_dict.keys()
space = product(*list(hp_dict.values()))

for subset in list(space):
    d = dict(zip(keys , subset))
    try:
        ret = run_trial(d)
    except Exception:
        traceback.print_exc()
        break
    print("Current settings:")
    print(json.dumps(d , indent=1))
    print("Results:")
    print(json.dumps(d, indent=1))

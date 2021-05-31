import os , argparse , logging
import utils
import torch
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    DataLoaderType, KGDataLoaderState
from tqdm import tqdm
from recbole.utils.utils import set_color
from recbole.config import Config
from recbole.data import data_preparation , create_dataset
from utils.data import add_graph
from recbole.utils import init_seed , init_logger
import commons, statics
from collections import defaultdict , OrderedDict
import pandas
import json

global_dict = OrderedDict()
all_metrics = ['recall' , 'ndcg' , 'precision' , 'hit']

def run_evaluation(model_name , dataset_name , model_path):

    global_dict[model_name] = OrderedDict()
    for metric in all_metrics:
        global_dict[model_name][metric] = OrderedDict()

    if dataset_name in ['ml-100k' , 'ml-1m']:
        kvals = [10,20,30]
    else:
        kvals = [5,10,20]

    for K in kvals:
        commons.init_seeds()
        model_class = statics.model_name_map[model_name]
        model_path = os.path.join("bestmodels" , dataset_name , "{}.pth".format(model_name))
        loaded_file = torch.load(model_path)
        config = loaded_file['config']
        config['data_path'] = os.path.join('dataset' , dataset_name)
        config['topk'] = K
        config['valid_metric'] = 'Recall@{}'.format(K)
        init_seed(config['seed'], config['reproducibility'])


        init_logger(config)
        logger = logging.getLogger()


        # dataset filtering
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        train_data = add_graph(train_data)


        model = model_class(config, train_data).to(commons.device)
        trainer = utils.get_trainer(config)(config, model)

        test_result = trainer.evaluate(test_data , load_best_model=True , model_file=model_path)
        for metric in all_metrics:
            global_dict[model_name][metric][K] = test_result["{}@{}".format(metric , K)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, action='store', help="dataset name")
    parser.add_argument("--models_path" , type = str , action='store' , help = "models_path")
    args, unknown = parser.parse_known_args()

    dataset_name = args.dataset
    mpath = args.models_path
    for model in ['ItemKNN' , 'BPR' ,  'NeuMF' , 'SpectralCF' , 'GCMC' , 'NGCF' , 'LightGCN' ]:
        run_evaluation(model , dataset_name , model_path = mpath)

    print(json.dumps(global_dict , indent = 2))

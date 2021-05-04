from run import run_trial
from collections import OrderedDict
import statics , commons
import argparse , traceback , json , os , pandas

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model" , type = str , action = 'store' , help = "model name")
    parser.add_argument("--dataset" , type = str , action = 'store' , help = "dataset name")
    args , unknown = parser.parse_known_args()


    model_name = args.model
    dataset_name = args.dataset

    from itertools import product

    hp_dict = statics.hyperparameters[args.model]
    keys = hp_dict.keys()
    space = product(*list(hp_dict.values()))
    entries = []

    docs = []
    metric_name = None
    for subset in list(space):
        params = OrderedDict(zip(keys , subset))
        try:
            ret = run_trial(model_name , dataset_name , hp_config=params)
        except Exception:
            traceback.print_exc()
            break
        metric_name = ret['metric']
        params[ ret['metric'] ] = ret['best_valid_score']
        docs.append(params)

    hp_df = pandas.DataFrame.from_records(docs)
    csv_name = os.path.join(commons.tuning_results_dir , model_name + '_' + dataset_name + '.csv')
    hp_df.sort_values(by = [metric_name] , inplace=True , ascending=False)
    hp_df.to_csv(csv_name)



"""
model_class = statics.model_name_map[model_name]

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
"""
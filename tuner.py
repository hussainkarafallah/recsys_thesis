from run import run_trial
from collections import OrderedDict
import statics , commons
import argparse , traceback , json , os , pandas
import multiprocessing

class Container(multiprocessing.Process):
    def __init__(self , queue , *args , **kwargs):
        super(Container, self).__init__(*args , **kwargs)
        self.queue = queue

    def run(self) -> None:
        ret = self._target(*self._args, **self._kwargs)
        print(os.getpid())
        self.queue.put(ret)

if __name__ == '__main__':
    #pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--model" , type = str , action = 'store' , help = "model name")
    parser.add_argument("--dataset" , type = str , action = 'store' , help = "dataset name")
    parser.add_argument("--file" , type = str , action = 'store' , help = "parameters file" )
    args , unknown = parser.parse_known_args()

    
    model_name = args.model
    dataset_name = args.dataset
    params_file = args.file

    from itertools import product

    with open(params_file) as f:
        hp_dict = json.load(f)
    hp_dict = hp_dict[args.model]
    print(hp_dict)
    keys = hp_dict.keys()
    space = product(*list(hp_dict.values()))
    entries = []

    docs = []
    metric_name = None
    q = multiprocessing.Queue()

    for subset in list(space):
        params = OrderedDict(zip(keys , subset))
        print(json.dumps(params, indent=1))
        container = Container(q, target=run_trial, args=(model_name,dataset_name) , kwargs={'hp_config' : params})
        container.start()
        container.join()
        ret = q.get()
        container.terminate()
        metric_name = ret['metric']
        params[ ret['metric'] ] = ret['best_valid_score']
        docs.append(params)


    hp_df = pandas.DataFrame.from_records(docs)
    csv_name = os.path.join(commons.tuning_results_dir , model_name + '_' + dataset_name + '.csv')
    hp_df.sort_values(by = [metric_name] , inplace=True , ascending=False)
    hp_df.to_csv(csv_name , index = False)

    #"""




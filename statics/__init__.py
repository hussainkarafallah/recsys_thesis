from .datasets_params import datasets_params
from training import *
from models import *
from recbole.trainer import Trainer

recbole_models = {
    'Pop',
    'BPR',
    'GCMC',
    'NGCF',
    'ItemKNN'
}

model_name_map = {
    'DeepWalk' : DeepWalk,
    'DeepWalk++' : ExtendedDeepWalk,
    'GCMC' : GCMC,
    'S-GCMC' : StochasticGCMC,
    'NGCF' : NGCF,
    'S-NGCF' : StochasticNGCF,
    'MF' : MF,
    'BPR' : BPR,
    'Pop' : Pop,
    'ItemKNN' : ItemKNN,
    'LightGCN' : LightGCN
}

trainers = {
    'DeepWalk' : StaticTrainer,
    'ExtendedDeepWalk' : Trainer,
    'StochasticGCMC' : StochasticTrainer,
    'StochasticNGCF' : StochasticTrainer,
    'LightGCN' : Trainer
}
from .datasets_params import datasets_params
from training import *
from models import *
from recbole.trainer import Trainer

recbole_models = {
    'Pop',
    'BPR',
    'GCMC',
    'NGCF'
}

model_name_map = {
    'DeepWalk' : DeepWalk,
    'DeepWalk++' : ExtendedDeepWalk,
    'GCMC' : GCMC,
    'S-GCMC' : StochasticGCMC,
    'NGCF' : NGCF,
    'MF' : MF,
    'BPR' : BPR,
    'Pop' : Pop
}

trainers = {
    'DeepWalk' : StaticTrainer,
    'DeepWalk++' : Trainer
}
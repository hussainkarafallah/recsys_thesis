hyperparameters = {
    'DeepWalk': {
        'num_walks' : [25 , 50 , 100],
        'walk_length' : [5 , 7 , 10],
        'window' : [5 , 10],
        'embeddings' : [64],
    },
    'DeepWalk++': {
        'num_walks' : [100 , 250],
        'walk_length' : [5 , 10 , 20],
        'window' : [5 , 10]
    }
}
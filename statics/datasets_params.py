
datasets_params = {
    'ml-100k' : {
        'min_user_inter_num' : 5,
        'min_item_inter_num' : 5,
        'eval_setting': 'TO_RS,uni100',
        'metrics' : ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': 20,
        'valid_metric' : 'Recall@20',
        "spilt_ratio" : [0.6,0.2,0.2],
    } ,
    'gowalla' : {
        'load_col' : {'inter': ['user_id', 'item_id','timestamp']},
        'min_user_inter_num' : 10,
        'min_item_inter_num' : 10,
        'eval_setting': 'TO_RS,uni100',
        'metrics' : ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': 20,
        'valid_metric' : 'Recall@20',
        'split_ratio' : [0.7 , 0.1 , 0.2]
    }
}

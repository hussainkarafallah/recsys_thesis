
datasets_params = {
    'ml-100k' : {
        'min_user_inter_num' : 5,
        'min_item_inter_num' : 5,
        'eval_setting': 'TO_RS,pop100',
        'metrics' : ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': 20,
        'valid_metric' : 'Recall@20'
    } ,
    'gowalla' : {
        'load_col' : {'inter': ['user_id', 'item_id','timestamp']},
        'min_user_inter_num' : 10,
        'min_item_inter_num' : 10,
        'eval_setting': 'TO_RS,pop100',
        'metrics' : ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': 20,
        'valid_metric' : 'Recall@20'
    }
}

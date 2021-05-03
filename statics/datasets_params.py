
datasets_params = {
    'ml-100k' : {
        'min_user_inter_num' : 5,
        'min_item_inter_num' : 5,
        'eval_setting': 'RO_RS,pop100',
        'metrics' : ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': 10,
        'valid_metric' : 'Recall@10'
    }
}

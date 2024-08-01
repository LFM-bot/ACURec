import argparse
from src.train.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', default='ACURec', type=str)
    parser.add_argument('--essential_ratio', default=0.5, type=float)
    parser.add_argument('--lamda', default=0.1, type=float)
    # Transformer
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--ffn_hidden', default=512, type=int, help='hidden dim for feed forward network')
    parser.add_argument('--num_blocks', default=2, type=int, help='number of transformer block')
    parser.add_argument('--num_heads', default=2, type=int, help='number of head for multi-head attention')
    parser.add_argument('--hidden_dropout', default=0.5, type=float, help='hidden state dropout rate')
    parser.add_argument('--attn_dropout', default=0.5, type=float, help='dropout rate for attention')
    parser.add_argument('--layer_norm_eps', default=1e-12, type=float, help='transformer layer norm eps')
    parser.add_argument('--initializer_range', default=0.02, type=float, help='transformer params initialize range')

    # Data
    parser.add_argument('--dataset', default='yelp', type=str)
    parser.add_argument('--max_len', default=50, type=int, help='max sequence length')
    # Training
    parser.add_argument('--epoch_num', default=400, type=int)
    parser.add_argument('--data_aug', action='store_false', help='data augmentation')
    parser.add_argument('--train_batch', default=2048, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--l2', default=0, type=float, help='l2 normalization')
    parser.add_argument('--patience', default=10, type=int, help='early stop patience')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--mark', default='', type=str)
    # Evaluation
    parser.add_argument('--split_type', default='valid_and_test', choices=['valid_only', 'valid_and_test'])
    parser.add_argument('--split_mode', default='LS', type=str, help='[LS, LS_R@0.x, PS]')
    parser.add_argument('--eval_mode', default='full', help='[uni100, pop100, full]')
    parser.add_argument('--k', default=[5, 10, 20, 50], help='rank k for each metric')
    parser.add_argument('--metric', default=['hit', 'ndcg'], help='[hit, ndcg, mrr, recall]')
    parser.add_argument('--valid_metric', default='hit@20', help='specifies which indicator to apply early stop')
    parser.add_argument('--do_test_with_eval', default=False, type=bool, help='do test with evaluation')

    config = parser.parse_args()

    trainer = Trainer(config)
    trainer.start_training()



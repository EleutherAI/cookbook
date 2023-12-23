# By Quentin Anthony and Beren Millidge

import argparse
import math

# Helper function to pretty-print message sizes
def convert_params(params):
    if params == 0:
        return "0"
    size_name = ("", "K", "M", "B", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(params, 1000)))
    p = math.pow(1000, i)
    s = round(params / p, 2)
    return "%s %s" % (s, size_name[i])

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", "-v",
                        type=int,
                        default=51200,
                        help='Size of the vocab')
    parser.add_argument("--hidden-size", "-hs",
                        type=int,
                        default=6144,
                        help='Dimension of the model\'s hidden size')
    parser.add_argument("--sequence-length", "-s",
                        type=int,
                        default=2048,
                        help='Sequence length used for training')
    parser.add_argument("--num-layers", "-l",
                        type=int,
                        default=44,
                        help='Number of transformer layers used in model')
    parser.add_argument("--moe",
                    action="store_true",
                    help='Whether our model is MoE')
    parser.add_argument("--num-experts", "-e",
                    type=int,
                    default=8,
                    help='Number of experts for MoE')
    parser.add_argument("--expert-interval", "-ei",
                    type=int,
                    default=1,
                    help='Expert interval for MoE')
    parser.add_argument("--topk", "-t",
                        type=int,
                        default=1,
                        help='Top k routing for MoE')
    parser.add_argument("--ffn-expansion-factor", "-ff",
                type=int,
                default=4,
                help='How much the MLP hidden size expands')
    parser.add_argument("--kv-size-ratio", "-kv",
                type=float,
                default=1.0,
                help='What fraction of num. query heads is num. key/value heads')
    return parser

# calculates the params of a model given their hparams
def calc_params(args):
    # Assumes that the embedding and unembedding are tied
    embedding_params = args.hidden_size * args.vocab_size
    position_embedding_params = args.hidden_size * args.sequence_length
    # Each QKVO matrix is (hxh)
    # Unless using GQA/MQA which makes K/V smaller
    attention_params = int(2 * (1 + args.kv_size_ratio) * args.num_layers * args.hidden_size * args.hidden_size)
    # (4*2)lh from the layernorm weights and biases for each of the QKV and mlp_in layernorms, 1h for the final layernorm.
    # the extra 4lh is a mystery but we include it here
    layernorm_params = 13 * args.num_layers * args.hidden_size
    #ffn_params = 12 * args.num_layers * args.hidden_size * args.hidden_size

    if args.moe:
        # the number of layers that are MoE. (e.g. interval is 2 for GShard)
        num_expert_layers = args.num_layers / args.expert_interval
        # the number of FFN params for each MoE layer
        ffn_expert_params = 2 * args.ffn_expansion_factor * num_expert_layers * args.num_experts * args.hidden_size * args.hidden_size
        # the number of FFN params for every dense layer
        ffn_dense_params = 2 * args.ffn_expansion_factor * (args.num_layers - num_expert_layers) * args.hidden_size * args.hidden_size
        ffn_params = ffn_expert_params + ffn_dense_params
        # the number of gating layer params assuming it's implemented as a simple linear layer
        gating_params = num_expert_layers * args.hidden_size * args.num_experts
    else:
        # two (h x [ffn_expansion_factor * h]) FFN matrices
        ffn_params = 2 * args.ffn_expansion_factor * args.num_layers * args.hidden_size * args.hidden_size

    total_params = embedding_params + attention_params + ffn_params + position_embedding_params + layernorm_params

    if args.moe:
        total_params += gating_params

    print(f'Calculating number of parameters with training configuration: {vars(args)}\n')
    print(f'Embedding parameters: {convert_params(embedding_params)}')
    print(f'Attention parameters: {convert_params(attention_params)}')
    print(f'FFN parameters: {convert_params(ffn_params)}')
    if args.moe:
        print(f'Gating parameters: {convert_params(gating_params)}')
    print(f'Total Params in the Model: {convert_params(total_params)}')

if __name__ == "__main__":
    print('\nExample with Fairseq-MoE 15B: python calc_transformer_params.py -l 12 -hs 768 --moe -e 512')
    print('Example with GPT-3 175B: python calc_transformer_params.py -l 96 -hs 12288')
    
    args = config_parser().parse_args()
    calc_params(args)

import argparse
from pylab import *
import pandas as pd

def read_transformer_logfile(logfile_name):
    all_values = []
    value_labels = ["num_attention_heads", "hidden_size", "train_micro_batch_size_per_gpu", "seq_length", "vocab_size", "train_batch_size", "tensor_mp_size", "pipeline_mp_size", "dp_size"]
    with open(logfile_name, 'r') as f:
        reading_estimate = False
        for line in f:
            line = line.strip()
            if line == "Estimate":
                reading_estimate = True
            elif line == "Actual":
                reading_estimate = False
            match = re.match(r'num_attention_heads: (\d+), hidden_size: (\d+), '
                             r'train_micro_batch_size_per_gpu: (\d+), seq_length: (\d+), '
                             r'vocab_size: (\d+), train_batch_size: (\d+), '
                             r'tensor_mp_size: (\d+), pipeline_mp_size: (\d+), '
                             r'dp_size: (\d+)', line)
            
            if match is not None:
                values = {}
                for i in range(1, 10):
                    values[value_labels[i-1]] = int(match.group(i))
                all_values.append(values)

            match = re.match(r'Throughput \(in TFLOP/s\) for qkv_transform \((.*)\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(2))
                if reading_estimate:
                    all_values[-1]["attention_key_value_query_transform"] = throughput
            
            match = re.match(r'Throughput \(in TFLOP/s\) for attention_score \((.*)\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(2))
                if reading_estimate:
                    all_values[-1]["attention_key_query_prob"] = throughput

            match = re.match(r'Throughput \(in TFLOP/s\) for attention_over_value \((.*)\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(2))
                if reading_estimate:
                    all_values[-1]["attention_prob_times_values"] = throughput

            match = re.match(r'Throughput \(in TFLOP/s\) for attention_dropout \((.*)\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(2))
                if reading_estimate:
                    all_values[-1]["attention_dropout"] = throughput
            
            match = re.match(r'Elapsed time for attention_softmax \((.*)\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(2))
                if reading_estimate:
                    all_values[-1]["attention_softmax"] = throughput

            match = re.match(r'Throughput \(in TFLOP/s\) for attention_linear_projection \((.*)\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(2))
                if reading_estimate:
                    all_values[-1]["attention_linear_projection"] = throughput 

            match = re.match(r'Throughput \(in TFLOP/s\) for mlp_h_to_4h \((.*)\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(2))
                if reading_estimate:
                    all_values[-1]["mlp_h_to_4h"] = throughput  

            match = re.match(r'Throughput \(in TFLOP/s\) for mlp_4h_to_h \((.*)\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(2))
                if reading_estimate:
                    all_values[-1]["mlp_4h_to_h"] = throughput 

            match = re.match(r'Elapsed time for mlp_fused_gelu \((.*)\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(2))
                if reading_estimate:
                    all_values[-1]["mlp_fused_gelu"] = throughput

            match = re.match(r'Elapsed time for transformer_add_bias_dropout \((.*)\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(2))
                if reading_estimate:
                    all_values[-1]["transformer_add_bias_dropout"] = throughput

            match = re.match(r'Elapsed time for transformer_layer_norm \((.*)\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(2))
                if reading_estimate:
                    all_values[-1]["transformer_layer_norm"] = throughput

            match = re.match(r'Throughput \(in TFLOP/s\) for logit_block \((.*)\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(2))
                if reading_estimate:
                    all_values[-1]["logit_block"] = throughput
                
            match = re.match(r'Attention duration \(in seconds\): (\d+\.\d+)', line)
            if match is not None:
                duration = float(match.group(1))
                if reading_estimate:
                    all_values[-1]["estimated_attention_duration"] = duration
                else:
                   all_values[-1]["actual_attention_duration"] = duration

            match = re.match(r'Attention throughput \(in TFLOP/s\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(1))
                if reading_estimate:
                    all_values[-1]["estimated_attention_throughput"] = throughput
                else:
                   all_values[-1]["actual_attention_throughput"] = throughput 
        
            match = re.match(r'MLP duration \(in seconds\): (\d+\.\d+)', line)
            if match is not None:
                duration = float(match.group(1))
                if reading_estimate:
                    all_values[-1]["estimated_mlp_duration"] = duration
                else:
                   all_values[-1]["actual_mlp_duration"] = duration

            match = re.match(r'MLP throughput \(in TFLOP/s\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(1))
                if reading_estimate:
                    all_values[-1]["estimated_mlp_throughput"] = throughput
                else:
                   all_values[-1]["actual_mlp_throughput"] = throughput

            match = re.match(r'Transformer duration \(in seconds\): (\d+\.\d+)', line)
            if match is not None:
                duration = float(match.group(1))
                if reading_estimate:
                    all_values[-1]["estimated_duration"] = duration
                else:
                    all_values[-1]["actual_duration"] = duration
            match = re.match(r'Transformer throughput \(in TFLOP/s\): (\d+\.\d+)', line)
            if match is not None:
                throughput = float(match.group(1))
                if reading_estimate:
                    all_values[-1]["estimated_throughput"] = throughput
                else:
                    all_values[-1]["actual_throughput"] = throughput
    return all_values

def read_mm_logfile(logfile_name):
    throughputs = []
    with open(logfile_name, 'r') as f:
        for line in f:
            line = line.strip()
            match = re.match(r'Throughput \(in TFLOP/s\) for (\d+)x(\d+)x(\d+): (\d+\.\d+)', line)
            if match is not None:
                m, n, k = int(match.group(1)), int(match.group(2)), int(match.group(3))
                throughput = float(match.group(4))
                throughputs.append({'m': m, 'n': n, 'k': k,
                                    'throughput': throughput})
    return throughputs

def read_bmm_logfile(logfile_name):
    throughputs = []
    with open(logfile_name, 'r') as f:
        for line in f:
            line = line.strip()
            match = re.match(r'Throughput \(in TFLOP/s\) for bmm \((\d+)x(\d+)x(\d+)x(\d+)\): (\d+\.\d+)', line)
            if match is not None:
                b, m, n, k = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
                throughput = float(match.group(5))
                throughputs.append({'b': b, 'm': m, 'n': n, 'k': k,
                                    'throughput': throughput})
    return throughputs

def to_pandas(filename):
    all_values_transformer = read_transformer_logfile(filename)
    all_values_mm = read_mm_logfile(filename)
    all_values_bmm = read_bmm_logfile(filename)
    if len(all_values_transformer) > 0:
        df = pd.DataFrame(all_values_transformer)
    elif len(all_values_mm) > 0:
        df = pd.DataFrame(all_values_mm)
    else:
        df = pd.DataFrame(all_values_bmm)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, help="Input log file")
    parser.add_argument("--output_file", type=str, help="Output csv file")
    args = parser.parse_args()
    df = to_pandas(args.file_name)
    df.to_csv(args.output_file, index=False)
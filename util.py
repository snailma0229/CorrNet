import torch
import numpy as np
import torch.nn as nn
from datasets import Seq2seq_Dataset_Kuai, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

def get_genral_vocab_dynamic_q(watch_ratio_all_for_vocab, q_start, q_end, q_decay_rate, epsilon):
    V = []
    cnt = []
    Y_sorted = np.sort(watch_ratio_all_for_vocab)[::-1]
    Y_updated = Y_sorted.copy()
    err = float('inf')
    q = q_start
    zi = float('1000')

    while err > epsilon and int(zi) != 0:
        zi = np.percentile(Y_updated, q * 100)
        if int(zi) == 0:
            break
        Y_updated = np.where(Y_updated < zi, Y_updated, Y_updated - zi)
        V.append(int(zi))
        cnt.append(Y_updated[Y_updated>=zi].shape[0])
        err = np.max(np.abs(Y_updated / (Y_sorted + 1e-10)))
        q = max(q_end, q * q_decay_rate)

    return V

def get_encoded_playtime(label_encoding_predict, begins, ends):
    temp_encoded_playtime = (ends - begins)

    encoded_prob = label_encoding_predict
    Ex_encoded_playtime = torch.sum(temp_encoded_playtime * encoded_prob, dim=-1, keepdim=True)
    e_x2 = torch.sum(torch.square(temp_encoded_playtime) * encoded_prob, dim=-1, keepdim=True)
    square_of_e_x = torch.square(Ex_encoded_playtime)
    var = torch.sqrt(e_x2 - square_of_e_x)
    return Ex_encoded_playtime, torch.sum(var)


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), seconds


def gamma_function(z, alpha):
    if alpha == 0:
        return z 
    return (1 - np.exp(-alpha * z)) / (1 - np.exp(-alpha))


def compute_bucket_edges(play_time, num_buckets, alpha):

    z_values = np.linspace(0, 1, num=num_buckets + 1)
    gamma_values = gamma_function(z_values, alpha)
    
    percentiles = gamma_values * 100
    bucket_edges = np.percentile(play_time, percentiles).tolist()

    begins_tensor = torch.tensor([bucket_edges[:-1]], dtype=torch.float32)
    ends_tensor = torch.tensor([bucket_edges[1:]], dtype=torch.float32)

    return begins_tensor, ends_tensor

def reduce_numbers(number, vocab):
    new_label = []
    remainder = number
    for value in vocab:
        while remainder >= value:
            remainder -= value
            new_label.append(value)
    return remainder, sorted(list(new_label), reverse=True)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def initialize_weights_xavier(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def initialize_transformer_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight.data)
        nn.init.zeros_(m.bias.data)

def initialize_weights_kaiming(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
        
def initialize_transformer_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias.data, -bound, bound)
    elif isinstance(m, nn.Embedding):
        nn.init.kaiming_uniform_(m.weight.data, a=math.sqrt(5))
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight.data)
        nn.init.zeros_(m.bias.data)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def create_intervals_and_loaders(features, labels, boundaries, batch_size, feat_num):
    
    datasets = {}
    data_loaders = {}

    for i in range(len(boundaries) + 1):
        if i == 0:
            indices = (features[:, -3] <= boundaries[i])
        elif i == len(boundaries):
            indices = (features[:, -3] > boundaries[i-1])
        else:
            indices = (features[:, -3] > boundaries[i-1]) & (features[:, -3] <= boundaries[i])

        selected_features = features[indices]
        selected_labels = [labels[idx] for idx in np.where(indices)[0].tolist()]

        dataset = Seq2seq_Dataset_Kuai(selected_features, selected_labels, is_train=False, feat_num=feat_num)
        data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

        datasets[f'interval{i}'] = dataset
        data_loaders[f'loader_interval{i}'] = data_loader

    return datasets, data_loaders


def count_matching_elements(pred_seqs, gt_seqs, num_elements):

    time_intervals = [0] * num_elements

    for index in range(len(pred_seqs)):
        gt_seq = gt_seqs[index]
        pred_seq = pred_seqs[index]
        max_check_length = min(len(gt_seq), len(pred_seq), num_elements)

        for i in range(max_check_length):
            if gt_seq[:i+1] == pred_seq[:i+1]:
                time_intervals[i] += 1
            else:
                break

    time_intervals = [round(num/len(pred_seqs), 4) for num in time_intervals]
    return time_intervals 


def label_process(vocab, numeric_vocab, len_train, watch_ratio_all_for_vocab, device):
    w2i_vocab = {word: idx for idx, word in enumerate(vocab)}
    i2w_vocab = {idx: word for idx, word in enumerate(vocab)}

    i2v_vocab = {}
    for idx, _ in i2w_vocab.items():
        if idx == 2 or idx == 0 or idx == 1:
            i2v_vocab[idx] = 0
        else:
            i2v_vocab[idx] = vocab[idx]
    vocab_size = len(i2v_vocab)
    vocab_values = torch.empty(vocab_size, dtype=torch.float32).to(device)
    for idx, value in i2v_vocab.items():
        vocab_values[idx] = value

    durations = []
    for i in vocab:
        if not isinstance(i, str):
            durations.append(i)
        else:
            durations.append(0)
    durations = torch.tensor(durations, device=device)

    watch_ratio_labels = []
    for index, watch_ratio in enumerate(tqdm(watch_ratio_all_for_vocab.tolist(), desc='Label processing')):
        watch_ratio_all_for_vocab[index], new_label = reduce_numbers(watch_ratio, numeric_vocab)
        label = ['<sos>'] + new_label + ['<eos>']
        watch_ratio_labels.append([vocab.index(i) for i in label])
    assert sum(watch_ratio_all_for_vocab != 0) == 0

    watch_ratio_labels_train = watch_ratio_labels[:len_train]
    watch_ratio_labels_test = watch_ratio_labels[len_train:]

    return durations, vocab_values, watch_ratio_labels_train, watch_ratio_labels_test
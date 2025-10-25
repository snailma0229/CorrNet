import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class InversePairsCalc:

    def InversePairs(self, data):
        
        if not data:
            return False
        if len(data)==1:
            return 0
        
        def merge(tuple_fir, tuple_sec):
            array_before = tuple_fir[0]
            cnt_before = tuple_fir[1]
            array_after = tuple_sec[0]
            cnt_after = tuple_sec[1]
            cnt = cnt_before + cnt_after
            flag = len(array_after) - 1
            array_merge = []
            for i in range(len(array_before)-1, -1, -1):
                while array_before[i]<=array_after[flag] and flag>=0 :
                    array_merge.append(array_after[flag])
                    flag -= 1
                if flag == -1 :
                    break
                else:
                    array_merge.append(array_before[i])
                    cnt += (flag+1)
            if flag == -1:
                for j in range(i,-1,-1):
                    array_merge.append(array_before[j])
            else:
                for j in range(flag ,-1,-1):
                    array_merge.append(array_after[j])
            return array_merge[::-1], cnt

        def mergesort(array):
            if len(array)==1:
                return (array,0)
            cut = math.floor(len(array)/2)
            tuple_fir=mergesort(array[:cut])
            tuple_sec=mergesort(array[cut:])
            return merge(tuple_fir, tuple_sec)
        return mergesort(data)[1]

def demo(a):
  S=InversePairsCalc()
  total_positive = S.InversePairs(a)
  print(total_positive)


def xauc_score(labels, pres):
    label_preds = zip(labels.reshape(-1), pres.reshape(-1))
    sorted_label_preds = sorted(
        label_preds, key=lambda lc: lc[1], reverse=True)
    label_preds_len = len(sorted_label_preds)
    pairs_cnt = label_preds_len * (label_preds_len-1) / 2
    
    labels_sort = [ele[0] for ele in sorted_label_preds]
    S=InversePairsCalc()
    total_positive = S.InversePairs(labels_sort)
    xauc = total_positive / pairs_cnt 
    return xauc


def embedding_constraint_loss(embeddings, vocab, start_id):
    embeddings = embeddings[start_id:, :]
    vocab = vocab[start_id:]
    loss = 0.0

    for i in range(0, embeddings.size(0), 2): 
        if i == embeddings.size(0)-1:
            break
        current = embeddings[i]
        next = embeddings[i+1]
        scale = vocab[i+1] / vocab[i]
        expected_next = current * scale
        loss += torch.norm(expected_next - next, p=2)

    # return loss / (embeddings.size(0) - 1) 
    return loss


def get_seq_huber_loss(predictions, targets):
    seq_len = predictions.shape[1]
    cumulative_losses = []

    for length in range(1, seq_len + 1):

        sub_predictions = predictions[:, :length] / 1000
        sub_targets = targets[:, :length] / 1000
        sub_losses = F.huber_loss(sub_predictions, sub_targets, reduction='sum')

        cumulative_losses.append(sub_losses)
    
    return sum(cumulative_losses)



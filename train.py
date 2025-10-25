import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model_zoo.transformer import Seq2Seq
from model_zoo.encoder import Encoder, EncoderWithMemoryAdaptors
from model_zoo.decoder import Decoder
from util import seconds_to_hms, init_weights, count_parameters, label_process, get_genral_vocab_dynamic_q
import numpy as np
import argparse
from metrics import xauc_score
from datasets import collate_fn_wt, Seq2seq_Dataset_Kuai_wt
import random
import os
import copy
import time
from sklearn.preprocessing import LabelEncoder
from inference import inference, inference_same_as_train, beam_search


def parse_args():
    parser = argparse.ArgumentParser(description="Set up model parameters (XLNet+DPO)")
    # for transformer
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_head', type=int, default=8, help='head num')
    parser.add_argument('--dec_layers', type=int, default=3, help='Number of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--ffn_dim', type=int, default=256, help='feedforward dimension')

    # for general setting
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--cls_weight', type=float, default=10.0, help='Class weight')
    parser.add_argument('--huber_weight', type=float, default=1.0, help='Huber loss weight')
    parser.add_argument('--pos_cls_weight', type=float, default=10)
    parser.add_argument('--neg_cls_weight', type=float, default=0.1)
    parser.add_argument('--log_dir', type=str, default='checkpoints_all/checkpoints_wr/', help='Log directory')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--train_npy', type=str, default='xxx/train.npy', help='Path to training features .npy')
    parser.add_argument('--test_npy', type=str, default='xxx/test.npy', help='Path to test features .npy')

    parser.add_argument('--use_binary_vocab', action='store_true')
    parser.add_argument('--use_dynamic_vocab', action='store_true')
    parser.add_argument('--use_manual_vocab', action='store_true')

    parser.add_argument('--q_start', type=float, default=0.9999)
    parser.add_argument('--q_end', type=float, default=0.9)
    parser.add_argument('--q_decay_rate', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=1e-6)

    parser.add_argument('--use_weight_input', action='store_true')
    parser.add_argument('--use_N_gram', action='store_true')
    parser.add_argument('--use_teacher_force', action='store_true')
    parser.add_argument('--teacher_force_ratio', type=float, default=0.5)

    parser.add_argument('--window_size', type=int, default=5, help='Window size')
    parser.add_argument('--use_memory_adaptors', action='store_true', help='Use position-conditioned memory adaptor encoder')
    parser.add_argument('--memory_adaptor_num', type=int, default=5, help='Number of memory adaptors (decoder positions)')
    parser.add_argument('--encoder_feature_num', type=int, default=2, help='Number of encoder feature groups per adaptor')

    # DPO-specific
    parser.add_argument('--dpo_last_k_epochs', type=int, default=3, help='Use DPO in last K epochs')
    parser.add_argument('--dpo_weight', type=float, default=3.0, help='DPO loss weight (when hybrid)')
    parser.add_argument('--dpo_beta', type=float, default=0.1, help='DPO beta scaling')
    parser.add_argument('--use_mse_threshold', action='store_true', help='Use MSE threshold negatives')
    parser.add_argument('--mse_threshold', type=float, default=5.0)
    parser.add_argument('--use_beam_search', action='store_true')
    parser.add_argument('--use_hybrid_train', action='store_true', help='Combine CE+Huber with DPO in DPO epochs')

    parser.add_argument('--resume_path', type=str, default='', help='Resume/best model path for negative generation')

    return parser.parse_args()


args = parse_args()
config = vars(args)
print("Model configuration:", config)

log_path = os.path.join(args.log_dir)
if not os.path.exists(log_path):
    os.makedirs(log_path)
save_path = os.path.join(args.log_dir, 'best_model.pth')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_feats_selected = np.load(args.train_npy)
test_feats_selected = np.load(args.test_npy)

watch_ratio_train = train_feats_selected[:, -4]
watch_ratio_test = test_feats_selected[:, -4]
watch_ratio_all = np.concatenate((watch_ratio_train, watch_ratio_test), axis=0)
watch_ratio_all_for_vocab = (watch_ratio_all * 1000).astype(int)

# label encoder for duration
print(train_feats_selected.shape)
print(test_feats_selected.shape)
train_durations = train_feats_selected[:, -3]
test_durations = test_feats_selected[:, -3]
durations_all = np.concatenate((train_durations, test_durations), axis=0)
encoder = LabelEncoder()
unique_sorted_durations = np.sort(np.unique(durations_all))
encoder.fit(unique_sorted_durations)
encoded_duration = encoder.transform(durations_all)
encoded_train_duration = encoded_duration[:len(train_feats_selected)]
encoded_test_duration = encoded_duration[len(train_feats_selected):]
train_feats_selected = np.column_stack((train_feats_selected, encoded_train_duration))
test_feats_selected = np.column_stack((test_feats_selected, encoded_test_duration))
print(train_feats_selected.shape)
print(test_feats_selected.shape)

feats_all = np.concatenate((train_feats_selected, test_feats_selected), axis=0)
max_values_per_column = np.amax(feats_all, axis=0)

special_tokens = ['<pad>', '<sos>', '<eos>']

if args.use_binary_vocab:
    numeric_vocab = [256, 128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.3, 0.1, 0.005, 0.003, 0.001]
    numeric_vocab = [i*1000 for i in numeric_vocab]
if args.use_manual_vocab:
    numeric_vocab = [150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 3, 1, 0.5, 0.3, 0.1, 0.005, 0.003, 0.001]
    numeric_vocab = [i*1000 for i in numeric_vocab]
if args.use_dynamic_vocab:
    numeric_vocab = get_genral_vocab_dynamic_q(watch_ratio_all_for_vocab, q_start=args.q_start, q_end=args.q_end, q_decay_rate=args.q_decay_rate, epsilon=args.epsilon)

vocab = special_tokens + numeric_vocab

# Prepare labels
durations, vocab_values, watch_ratio_labels_train, watch_ratio_labels_test = label_process(vocab, numeric_vocab, len(train_feats_selected), watch_ratio_all_for_vocab, device)
assert len(watch_ratio_labels_train) == train_feats_selected.shape[0]
max_len_position = max(max([len(i) for i in watch_ratio_labels_train]), max([len(j) for j in watch_ratio_labels_test]))

train_dataset = Seq2seq_Dataset_Kuai_wt(train_feats_selected, watch_ratio_labels_train)
test_dataset = Seq2seq_Dataset_Kuai_wt(test_feats_selected, watch_ratio_labels_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn_wt, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_wt, shuffle=False)

if args.use_memory_adaptors:
    encoder = EncoderWithMemoryAdaptors(enconder_fea_num=args.encoder_feature_num,
                                        memory_adaptor_num=args.memory_adaptor_num,
                                        input_dim=args.feat_dim,
                                        hidden_dim=args.hidden_dim,
                                        dropout_rate=args.dropout)
else:
    encoder = Encoder(input_dim=args.feat_dim, hidden_dim=args.hidden_dim, dropout_rate = args.dropout)

decoder = Decoder(d_model=args.hidden_dim,
                  n_head=args.n_head,
                  ffn_hidden=args.ffn_dim,
                  dec_voc_size=len(vocab),
                  drop_prob=args.dropout,
                  n_layers=args.dec_layers,
                  max_len=max_len_position,
                  device=device)

model = Seq2Seq(encoder, decoder, args.use_weight_input, args.use_N_gram,
                args.use_teacher_force, args.teacher_force_ratio,
                max_values_per_column, args.feat_dim, device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(init_weights)

embedding_params = []
other_params = []
for name, param in model.named_parameters():
    if 'embedding' in name:
        embedding_params.append(param)
    else:
        other_params.append(param)

optimizer = torch.optim.Adam([
    {'params': embedding_params, 'weight_decay': 1e-4}, 
    {'params': other_params, 'weight_decay': 0} 
], lr=args.lr) 

criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

best_mae = float('inf')
best_xauc = 0
best_model_state = None
start_time = time.time()
print("vocab: ", vocab)

# Helper to load best model for DPO negative sampling

def load_best_model_for_eval(path_or_best_state, model):
    m = copy.deepcopy(model)
    if isinstance(path_or_best_state, str) and path_or_best_state:
        m.load_state_dict(torch.load(path_or_best_state, map_location=device))
    else:
        m.load_state_dict(path_or_best_state)
    m.to(device)
    m.eval()
    return m

# Determine DPO start epoch
if args.dpo_last_k_epochs > 0:
    dpo_start_epoch = max(1, args.num_epochs - args.dpo_last_k_epochs + 1)
else:
    dpo_start_epoch = args.num_epochs + 1  # never

################################ Train ################################

for epoch in range(1, args.num_epochs+1):
    epoch_start = time.time()

    model.train()
    epoch_loss = 0

    use_dpo_epoch = (epoch >= dpo_start_epoch)

    for num_step, (X_batch, qmsk_batch, imsk_batch, play_times, Seq_label_batch, Seq_label_mask) in enumerate(train_loader, 1):
        X_batch = X_batch.to(device)
        qmsk_batch = qmsk_batch.to(device)
        imsk_batch = imsk_batch.to(device)
        Ytime_batch = play_times.to(device)
        Seq_label_batch = Seq_label_batch.to(device=device, dtype=torch.long)
        Seq_label_mask = Seq_label_mask.to(device)

        optimizer.zero_grad()

        if not use_dpo_epoch:
            # XLNet-style supervised training
            output, content_output, masked_tokens = model(X_batch, qmsk_batch, imsk_batch, Seq_label_batch, Seq_label_mask)

            output_softmax = F.softmax(output, dim=-1)
            output_dim = output.shape[-1]
            seq_len = output.shape[1]

            argmax_indices = output_softmax.argmax(dim=-1)
            indices = torch.arange(output_dim, device=output.device).unsqueeze(0).unsqueeze(0).expand(output.shape[0], seq_len, -1)
            start_indices = argmax_indices.unsqueeze(-1) - args.window_size // 2
            end_indices = argmax_indices.unsqueeze(-1) + args.window_size // 2 + 1
            mask = (indices >= start_indices) & (indices < end_indices)

            weighted_durations = torch.sum(output_softmax * mask * durations.unsqueeze(0).unsqueeze(0), dim=-1)
            weighted_durations = weighted_durations * Seq_label_mask[:, 1:]
            pre_watch_times = torch.sum(weighted_durations, dim=1) / 1000

            logits = output.contiguous().view(-1, output_dim)
            trg = Seq_label_batch[:,1:].contiguous().view(-1)

            loss_ce = criterion(logits, trg)
            loss_huber = F.huber_loss(Ytime_batch.squeeze(), pre_watch_times, reduction='sum')

            positive_logits = content_output[masked_tokens]
            negative_logits = content_output[~masked_tokens]
            positive_targets = Seq_label_batch[:,1:][masked_tokens]
            negative_targets = Seq_label_batch[:,1:][~masked_tokens]
            positive_loss =  criterion(positive_logits, positive_targets)
            negative_loss = criterion(negative_logits, negative_targets)

            loss = args.cls_weight * loss_ce + args.huber_weight * loss_huber \
                + args.pos_cls_weight * positive_loss + args.neg_cls_weight * negative_loss
        else:
            # DPO training using XLNet outputs
            # teacher input is Seq_label_batch, output is distributions for positions 1..seq
            output, content_output, masked_tokens = model(X_batch, qmsk_batch, imsk_batch, Seq_label_batch, Seq_label_mask)
            output_dim = output.shape[-1]

            # Positives
            pos_samples = Seq_label_batch[:, 1:]  # (batch, seq-1)
            pos_log_probs = F.log_softmax(output, dim=-1).gather(dim=-1, index=pos_samples.unsqueeze(-1)).squeeze(-1)
            pos_log_probs = pos_log_probs * Seq_label_mask[:, 1:]
            pos_log_probs = torch.clamp(pos_log_probs, min=-100)
            pos_log_prob = pos_log_probs.sum(dim=-1)  # (batch,)

            # Build best_model for negatives
            with torch.no_grad():
                if best_model_state is not None:
                    best_ref = best_model_state
                elif os.path.isfile(save_path):
                    best_ref = save_path
                elif args.resume_path:
                    best_ref = args.resume_path
                else:
                    best_ref = None
                best_model = load_best_model_for_eval(best_ref, model)

                if args.use_mse_threshold:
                    best_output, _, _ = best_model(X_batch, qmsk_batch, imsk_batch, Seq_label_batch, Seq_label_mask)
                    best_softmax = F.softmax(best_output, dim=-1)
                    best_pred = best_softmax.argmax(dim=-1)

                    best_output_dim = best_output.shape[-1]
                    best_seq_len = best_output.shape[1]
                    best_argmax_indices = best_softmax.argmax(dim=-1)
                    best_indices = torch.arange(best_output_dim, device=best_output.device).unsqueeze(0).unsqueeze(0).expand(best_output.shape[0], best_seq_len, -1)
                    best_start_indices = best_argmax_indices.unsqueeze(-1) - args.window_size // 2
                    best_end_indices = best_argmax_indices.unsqueeze(-1) + args.window_size // 2 + 1
                    best_mask = (best_indices >= best_start_indices) & (best_indices < best_end_indices)

                    best_weighted_durations = torch.sum(best_softmax * best_mask * durations.unsqueeze(0).unsqueeze(0), dim=-1)
                    best_weighted_durations = best_weighted_durations * Seq_label_mask[:, 1:]
                    best_pre_watch_times = torch.sum(best_weighted_durations, dim=1) / 1000
                    mse = torch.abs((best_pre_watch_times - Ytime_batch.squeeze()))
                    neg_mask = mse > args.mse_threshold

                if args.use_beam_search:
                    beam_sequences = beam_search(best_model, X_batch, qmsk_batch, imsk_batch, seq_len=Seq_label_batch.shape[1], device=device)
                    max_mse = -float('inf')
                    neg_samples_2 = None
                    for beam_seq in beam_sequences:
                        with torch.no_grad():
                            beam_values_tensor = vocab_values[beam_seq.long()]
                            beam_two_first_occurrences = (beam_seq == 2).long().argmax(dim=1)
                            beam_row_length = beam_seq.size(1)
                            beam_two_first_occurrences += ((beam_seq == 2).sum(dim=1) == 0).long() * beam_row_length
                            beam_range_indices = torch.arange(beam_row_length).expand_as(beam_seq).to(device)
                            beam_mask = beam_range_indices < beam_two_first_occurrences.unsqueeze(1)
                            beam_values_tensor *= beam_mask.float()
                            beam_pre_watch_time = beam_values_tensor.sum(dim=1) / 1000
                            beam_mse = torch.mean((beam_pre_watch_time - Ytime_batch.squeeze()) ** 2).item()
                        if beam_mse > max_mse:
                            max_mse = beam_mse
                            neg_samples_2 = beam_seq[:, 1:]

            # Negatives
            if args.use_mse_threshold:
                neg_log_probs_1 = F.log_softmax(output, dim=-1).gather(dim=-1, index=best_pred.unsqueeze(-1)).squeeze(-1)
                neg_log_probs_1 = neg_log_probs_1 * Seq_label_mask[:, 1:]
                neg_log_probs_1 = torch.clamp(neg_log_probs_1, min=-100)
                neg_log_prob_1 = neg_log_probs_1.sum(dim=-1)
                neg_log_prob_1 = neg_log_prob_1 * neg_mask.float()

            if args.use_beam_search and 'neg_samples_2' in locals() and neg_samples_2 is not None:
                neg_log_probs_2 = F.log_softmax(output, dim=-1).gather(dim=-1, index=neg_samples_2.unsqueeze(-1)).squeeze(-1)
                neg_log_probs_2 = neg_log_probs_2 * Seq_label_mask[:, 1:]
                neg_log_probs_2 = torch.clamp(neg_log_probs_2, min=-100)
                neg_log_prob_2 = neg_log_probs_2.sum(dim=-1)

            beta = args.dpo_beta
            if args.use_mse_threshold:
                loss_dpo = -torch.log(torch.sigmoid(beta * (pos_log_prob - neg_log_prob_1)))
                if args.use_beam_search and 'neg_log_prob_2' in locals():
                    loss_dpo = loss_dpo + -torch.log(torch.sigmoid(beta * (pos_log_prob - neg_log_prob_2)))
            elif args.use_beam_search and 'neg_log_prob_2' in locals():
                loss_dpo = -torch.log(torch.sigmoid(beta * (pos_log_prob - neg_log_prob_2)))
            else:
                loss_dpo = -pos_log_prob
            loss_dpo = loss_dpo.sum()

            if args.use_hybrid_train:
                output_softmax = F.softmax(output, dim=-1)
                seq_len = output.shape[1]

                argmax_indices = output_softmax.argmax(dim=-1)
                indices = torch.arange(output_dim, device=output.device).unsqueeze(0).unsqueeze(0).expand(output.shape[0], seq_len, -1)
                start_indices = argmax_indices.unsqueeze(-1) - args.window_size // 2
                end_indices = argmax_indices.unsqueeze(-1) + args.window_size // 2 + 1
                mask = (indices >= start_indices) & (indices < end_indices)

                weighted_durations = torch.sum(output_softmax * mask * durations.unsqueeze(0).unsqueeze(0), dim=-1)
                weighted_durations = weighted_durations * Seq_label_mask[:, 1:]
                pre_watch_times = torch.sum(weighted_durations, dim=1) / 1000

                logits = output.contiguous().view(-1, output_dim)
                trg = Seq_label_batch[:,1:].contiguous().view(-1)
                loss_ce = criterion(logits, trg)
                loss_huber = F.huber_loss(Ytime_batch.squeeze(), pre_watch_times, reduction='sum')
                loss = args.cls_weight * loss_ce + args.huber_weight * loss_huber + args.dpo_weight * loss_dpo
            else:
                loss = loss_dpo

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if num_step == 1 or num_step % 2000 == 0:
            if not use_dpo_epoch:
                print("Epoch: {}/{}, Step: {}, Supervised Loss: {:.4f}".format(epoch, args.num_epochs, num_step, loss.item()))
            else:
                print("Epoch: {}/{}, Step: {}, DPO Loss: {:.4f}".format(epoch, args.num_epochs, num_step, loss_dpo.item()))

    print(f"{epoch}/{args.num_epochs} Optimization Finished!")
    all_pred_list = []
    all_ground_list = []
    model.eval()
    max_indices, gt_times = inference(test_loader, model, args.use_weight_input, args.use_N_gram, device)
    values_tensor = vocab_values[max_indices.long()]

    two_first_occurrences = (max_indices == 2).long().argmax(dim=1)
    row_length = max_indices.size(1)
    two_first_occurrences += ((max_indices == 2).sum(dim=1) == 0).long() * row_length
    range_indices = torch.arange(row_length).expand_as(max_indices).to(device)
    mask = range_indices < two_first_occurrences.unsqueeze(1)
    values_tensor *= mask.float() 

    pre_watch_times = values_tensor.sum(dim=1) / 1000

    all_pred_list.append(pre_watch_times.cpu().numpy())
    all_pred = np.concatenate(all_pred_list).reshape(-1)

    all_ground_list.append(gt_times.cpu().numpy())
    all_yy = np.concatenate(all_ground_list).reshape(-1)

    xauc = round(xauc_score(all_yy, all_pred), 4)
    print(f"{epoch}/{args.num_epochs} epoch XAUC is {xauc}")
    mae = round(np.mean(np.abs(all_pred - all_yy)), 4)
    print(f"{epoch}/{args.num_epochs} epoch MAE is {mae}")

    ckpt_path = os.path.join(args.log_dir, f'epoch_{epoch}.pth')
    ckpt_model = copy.deepcopy(model.state_dict())
    torch.save(ckpt_model, ckpt_path)

    if mae < best_mae:
        best_mae = mae
        best_xauc = xauc
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, save_path)
    
    epoch_end = time.time()
    elapsed_time = epoch_end - start_time
    remaining_epochs = args.num_epochs - epoch
    average_time_per_epoch = elapsed_time / epoch
    estimated_remaining_time = remaining_epochs * average_time_per_epoch
    elapsed_hours, elapsed_minutes, elapsed_seconds = seconds_to_hms(elapsed_time)
    remaining_hours, remaining_minutes, remaining_seconds = seconds_to_hms(estimated_remaining_time)
    print(f"Elapsed Time: {elapsed_hours}h {elapsed_minutes}m {elapsed_seconds:.2f}s, Est Time: {remaining_hours}h {remaining_minutes}m {remaining_seconds:.2f}s")

    print(f"Best XAUC is {best_xauc} \nBest MAE is {best_mae}")

    inference_same_as_train(ckpt_path, test_loader, model, args.window_size, args.use_weight_input, args.use_N_gram, durations, device)

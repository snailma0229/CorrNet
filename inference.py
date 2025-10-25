import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from metrics import xauc_score

def inference(test_loader, model, use_weight_input, use_N_gram, device, max_len = 20):
    max_indices_list = []
    gt_times_list = []
    with torch.no_grad():
        for X_batch, qmsk_batch, imsk_batch, play_times, Seq_label_batch, Seq_label_mask in tqdm(test_loader, desc="Testing Progress"):
            X_batch = X_batch.to(device)
            qmsk_batch = qmsk_batch.to(device)
            imsk_batch = imsk_batch.to(device)
            Ytime_batch = play_times.to(device)
            Seq_label_batch = Seq_label_batch.to(device=device, dtype=torch.long).transpose(0,1)
            Seq_label_mask = Seq_label_mask.to(device)

            enc_src, src_mask = model.get_encoder_result(X_batch, qmsk_batch, imsk_batch)

            input = Seq_label_batch[0, :].unsqueeze(1)
            
            if use_weight_input:
                new_token = torch.zeros_like(input)
                query = None
                for step in range(max_len - 1):
                    input = torch.cat([input, new_token], dim=-1)
                    output, content_output, q, c = model.decoder.inference_weight_forward(input, enc_src, src_mask, use_weight_input, step, query)
                    prob = F.softmax(output, dim=-1)
                    next_word = torch.argmax(prob, dim=-1).unsqueeze(-1)
                    input[:, -1:] = next_word
                    query = output
            elif use_N_gram:
                new_token = torch.zeros_like(input)
                query = None
                for step in range(max_len - 1):
                    input = torch.cat([input, new_token], dim=-1)
                    output, content_output, q, c = model.decoder.inference_weight_forward(input, enc_src, src_mask, use_weight_input, step, query)
                    prob = F.softmax(output, dim=-1)
                    next_word = torch.argmax(prob, dim=-1).unsqueeze(-1)
                    input[:, -1:] = next_word
                    query = q
            else:
                new_token = torch.zeros_like(input)
                for _ in range(max_len - 1):
                    input = torch.cat([input, new_token], dim=-1)
                    output, content_output, q, c = model.decoder.inference(input, enc_src, src_mask)
                    prob = F.softmax(output, dim=-1)
                    next_word = torch.argmax(prob, dim=-1).unsqueeze(-1)
                    input[:, -1:] = next_word

            max_indices_list.append(input.tolist())
            gt_times_list.append(Ytime_batch.squeeze().tolist())

    max_indices = torch.tensor([item for sublist in max_indices_list for item in sublist], dtype=torch.int64).to(device).view(-1, max_len)
    gt_times = torch.tensor([item for sublist in gt_times_list for item in sublist], dtype=torch.float64).to(device).view(-1)

    return max_indices, gt_times


def inference_same_as_train(save_path, test_loader, model, 
                            window_size, use_weight_input,
                            use_N_gram, durations, device, max_len = 20):
    
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    
    print("Inference the same as Train...")

    gt_times_list = []
    all_pred_list = []
    all_ground_list = []
    with torch.no_grad():
        for X_batch, qmsk_batch, imsk_batch, play_times, Seq_label_batch, Seq_label_mask in test_loader:
            X_batch = X_batch.to(device)
            qmsk_batch = qmsk_batch.to(device)
            imsk_batch = imsk_batch.to(device)
            Ytime_batch = play_times.to(device)
            Seq_label_batch = Seq_label_batch.to(device=device, dtype=torch.long).transpose(0,1)
            Seq_label_mask = Seq_label_mask.to(device)

            enc_src, src_mask = model.get_encoder_result(X_batch, qmsk_batch, imsk_batch)

            input = Seq_label_batch[0, :].unsqueeze(1)        
            output_prob = []
            if use_weight_input:
                new_token = torch.zeros_like(input)
                query = None
                for step in range(max_len - 1):
                    input = torch.cat([input, new_token], dim=-1)
                    output, content_output, q, c = model.decoder.inference_weight_forward(input, enc_src, src_mask, use_weight_input, step, query) # batch, seq, vocab
                    prob = F.softmax(output, dim=-1) # batch, vocab
                    next_word = torch.argmax(prob, dim=-1).unsqueeze(-1)  # batch, 1
                    input[:, -1:] = next_word
                    query = output
                    output_prob.append(prob)
            elif use_N_gram:
                new_token = torch.zeros_like(input)
                query = None
                for step in range(max_len - 1):
                    input = torch.cat([input, new_token], dim=-1)
                    output, content_output, q, c = model.decoder.inference_weight_forward(input, enc_src, src_mask, use_weight_input, step, query) # batch, seq, vocab
                    prob = F.softmax(output, dim=-1) # batch, vocab
                    next_word = torch.argmax(prob, dim=-1).unsqueeze(-1)  # batch, 1
                    input[:, -1:] = next_word
                    query = q
                    output_prob.append(prob)
            else:
                new_token = torch.zeros_like(input)
                for _ in range(max_len - 1):
                    input = torch.cat([input, new_token], dim=-1)
                    output, content_output, q, c = model.decoder.inference(input, enc_src, src_mask) # batch, seq, vocab
                    prob = F.softmax(output, dim=-1) # batch, vocab
                    next_word = torch.argmax(prob, dim=-1).unsqueeze(-1)  # batch, 1
                    input[:, -1:] = next_word
                    output_prob.append(prob)

            output_prob = [i.unsqueeze(1) for i in output_prob]
            outputs = torch.cat(output_prob, dim=1) # (batch, 19, vocab)
            max_indices = torch.tensor([i for i in input.tolist()], dtype=torch.int64).to(device).view(-1, max_len)
            gt_times_list = torch.tensor(Ytime_batch.squeeze().tolist(), dtype=torch.float64).to(device).view(-1)

            two_first_occurrences = (max_indices == 2).long().argmax(dim=1)
            row_length = max_indices.size(1)
            two_first_occurrences += ((max_indices == 2).sum(dim=1) == 0).long() * row_length
            range_indices = torch.arange(row_length).expand_as(max_indices).to(device)
            mask = range_indices <= two_first_occurrences.unsqueeze(1)

            # window
            argmax_indices = outputs.argmax(dim=-1)
            indices = torch.arange(outputs.shape[2], device=outputs.device).unsqueeze(0).unsqueeze(0).expand(outputs.shape[0], outputs.shape[1], -1)
            start_indices = argmax_indices.unsqueeze(-1) - window_size // 2 # batch, seq, 1
            end_indices = argmax_indices.unsqueeze(-1) + window_size // 2 + 1 # batch, seq, 1
            mask_batch = (indices >= start_indices) & (indices < end_indices)

            weighted_durations = torch.sum(outputs * mask_batch * durations.unsqueeze(0).unsqueeze(0), dim=-1)
            weighted_durations *= mask[:, 1:]
            pre_watch_times = torch.sum(weighted_durations, dim=1) / 1000

            preds = pre_watch_times.cpu().numpy()
            gts = gt_times_list.cpu().numpy()

            all_pred_list.append(preds)
            all_ground_list.append(gts)

    all_pred = np.concatenate(all_pred_list).reshape(-1)
    all_yy = np.concatenate(all_ground_list).reshape(-1)
    xauc = round(xauc_score(all_yy, all_pred), 4)  
    mae = round(np.mean(np.abs(all_pred - all_yy)), 4)
    print(f"XAUC is {xauc} \nMAE is {mae}")

def beam_search(model, X_batch, qmsk_batch, imsk_batch, seq_len, beam_size=5, device='cuda'):
    model.eval()
    batch_size = X_batch.shape[0]

    with torch.no_grad():
        enc_src, src_mask = model.get_encoder_result(X_batch, qmsk_batch, imsk_batch)
    initial_input = torch.ones((batch_size, 1), dtype=torch.long, device=device)
    beams = [(initial_input, 0.0)]

    for _ in range(seq_len - 1):
        new_beams = []
        for seq, log_prob in beams:
            with torch.no_grad():
                output, content_output, q, c = model.decoder.inference(seq, enc_src, src_mask)
                output_softmax = F.softmax(output, dim=-1)
                log_probs = torch.log(output_softmax)
                topk_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)
            for k in range(beam_size):
                next_token = topk_indices[:, k].unsqueeze(-1)
                new_seq = torch.cat([seq, next_token], dim=-1)
                new_prob = log_prob + topk_probs[:, k].mean().item()
                new_beams.append((new_seq, new_prob))
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
    return [seq for seq, _ in beams]
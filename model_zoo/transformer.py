import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, use_weight_input, use_N_gram,
                 use_teacher_force, teacher_force_ratio,
                 max_values_per_column, feat_dim, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.use_weight_input = use_weight_input
        self.use_N_gram = use_N_gram
        self.use_teacher_force = use_teacher_force
        self.teacher_force_ratio = teacher_force_ratio
        assert not (self.use_N_gram and self.use_weight_input), "use_weight_input and use_N_gram must not be True at the same time"

        self.user_id_embedding = nn.Embedding(int(max_values_per_column[0])+1, feat_dim)
        self.user_fea_embedding = nn.ModuleList([
            nn.Embedding(int(max_values_per_column[i])+1, feat_dim) for i in range(1, 19)
        ])
        self.item_id_embedding = nn.Embedding(int(max_values_per_column[19])+1, feat_dim)
        self.item_fea_embedding = nn.Embedding(int(max(max_values_per_column[20:24]))+1, feat_dim)
        self.item_duration_embedding = nn.Embedding(int(max_values_per_column[-1])+1, feat_dim)

        self.device = device
    
    def make_trg_mask(self, trg, trg_pad_mask):
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        trg_mask = trg_pad_mask.unsqueeze(1) & trg_sub_mask.unsqueeze(0)

        return trg_mask.unsqueeze(1)
    
    def get_encoder_result(self, x, qmsk, imsk):

        usr_id = self.user_id_embedding(x[:, 0].unsqueeze(1))
        usr_fea = torch.cat([self.user_fea_embedding[id](x[:, i].unsqueeze(1)) for id, i in enumerate(range(1,19))], dim=1)
        item_id = self.item_id_embedding(x[:, 19].unsqueeze(1))
        item_fea = torch.cat([self.item_fea_embedding(x[:, i].unsqueeze(1)) for i in range(20,24)], dim=1)
        item_duration = self.item_duration_embedding(x[:, 24].unsqueeze(1))

        usr_fea = torch.cat([usr_id, usr_fea], dim=1)
        item_fea = torch.cat([item_id, item_fea], dim=1)

        qmsk = qmsk.unsqueeze(-1)
        imsk = imsk.unsqueeze(-1)
        
        query_embeddings = usr_fea * qmsk
        item_embeddings = item_fea * imsk
        item_embeddings = torch.cat([item_embeddings, item_duration], dim=1)
        
        query_sum = query_embeddings.sum(1) / qmsk.sum(1)
        item_sum = item_embeddings.sum(1) / (imsk.sum(1) + 1)
        src_input = torch.cat([query_sum, item_sum], dim=1)

        enc_out = self.encoder(src_input)
        # Support both standard (1, batch, dim) and memory-adaptor (batch, mem_num, enc_num, dim)
        if enc_out.dim() == 4:
            # (batch, mem_num, enc_num, dim)
            enc_src = enc_out
            src_len = enc_src.shape[2]
        elif enc_out.dim() == 3:
            # (1, batch, dim) -> (batch, 1, dim)
            enc_src = enc_out.transpose(0,1)
            src_len = enc_src.shape[1]
        else:
            # (batch, dim) -> (batch, 1, dim)
            enc_src = enc_out.unsqueeze(1)
            src_len = 1

        src_mask = torch.ones(qmsk.shape[0], src_len).to(self.device).unsqueeze(1).unsqueeze(2)

        return enc_src, src_mask
    
    def forward(self, x, qmsk, imsk, trg, true_label_mask):

        enc_src, src_mask = self.get_encoder_result(x, qmsk, imsk)

        # trg (batch, seq)
        # enc_src (batch, 1, dim)
        # src_mask (batch, 1, 1, seq_src(1))
        query_output, content_output, query, content = self.decoder(trg, enc_src, true_label_mask, src_mask)

        if not self.use_teacher_force:
            masked_tokens = (trg[:, 1:] != trg[:, 1:]).bool()
            
        else:  
            sample_mask = (torch.rand(true_label_mask[:, 1:].size()) > self.teacher_force_ratio).long().to(query_output.device)
            sample_mask = (sample_mask & true_label_mask[:, 1:])
            masked_tokens = sample_mask.bool()
            # sample from predictions
            k = 3
            size = trg[:, 1:].size()
            _, prob = query_output.topk(k, dim=-1)
            random_choices = torch.randint(0, k, (size[0] * size[1],), dtype=torch.long)
            offsets = torch.arange(0, size[0] * size[1] * k, k, dtype=torch.long)
            choice = (random_choices + offsets).to(trg)
            prob = prob.reshape(-1)[choice].reshape(size)
            teacher_input = prob * sample_mask + trg[:, 1:] * (1-sample_mask)
            teacher_input = torch.cat([trg[:, :1], teacher_input], dim=-1)

            if self.use_weight_input:
                query_output, content_output, _, _ = self.decoder.weight_forward(teacher_input, enc_src, 
                                                                                 true_label_mask, src_mask, 
                                                                                 self.use_weight_input,
                                                                                 masked_tokens, query_output)
            elif self.use_N_gram:
                query_output, content_output, _, _ = self.decoder.weight_forward(teacher_input, enc_src, 
                                                                                 true_label_mask, src_mask, 
                                                                                 self.use_weight_input,
                                                                                 masked_tokens, query)
            else:
                query_output, content_output, _, _ = self.decoder(teacher_input, enc_src, true_label_mask, src_mask)
           
            # prob = query_output.argmax(dim=-1) # (batch_size, seq) index

            # for i in range(output.shape[1]):
            #     if teacher_force[i-1] and i != 0:
            #         if self.use_weight_input:
            #             input_teacher.append(output[:,i-1,:]) # (batch_size, token_num)
            #         elif self.use_N_gram:
            #             input_teacher.append(output_fea[:,i-1,:]) # (batch_size, dim)
            #         else:
            #             input_teacher.append(prob[:,i-1].unsqueeze(1)) # batch, 1
            #     else:
            #         input_teacher.append(trg[:,i].unsqueeze(1))

            # if not self.use_weight_input and not self.use_N_gram:
            #     input_teacher = torch.cat(input_teacher, dim=1)

            # output, _ = self.decoder.weight_forward(input_teacher, enc_src, trg_mask, src_mask, 
            #                                      self.use_weight_input, self.use_N_gram)

        return query_output, content_output, masked_tokens
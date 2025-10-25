import torch
import torch.nn as nn
import math
from model_zoo.attention import PositionwiseFeedForward, MultiHeadAttention
import torch.nn.functional as F
from model_zoo.position_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, max_len, device):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

        self.tok_emb = nn.Embedding(dec_voc_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, math.ceil(max_len / 100) * 100, device)
        self.drop_out = nn.Dropout(p=drop_prob)
        self.output_dim = dec_voc_size

    def buffered_future_mask(self, tensor):
        dim = tensor.size(1)
        return torch.triu(torch.full((dim, dim), float('-inf')), 1).to(tensor)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):

        # trg (batch, seq)
        # enc_src (batch, 1, dim)
        # trg_mask (batch, seq)
        # src_mask (batch, 1, 1, seq_src(1))

        sz = trg.size() # (batch, seq_len_tgt)
        positions = self.pos_emb(trg) # (batch, seq_len_tgt, dim)

        q = trg.new_full(trg.size(), 0)  # (batch, seq_len_tgt)
        q = self.drop_out(self.tok_emb(q) + positions) # (batch, seq_len_tgt, dim)

        c = self.drop_out(self.tok_emb(trg) + positions) # (batch, seq_len_tgt, dim)
        self_attn_mask = self.buffered_future_mask(c)

        for layer in self.layers:
            q, c = layer(q, c, enc_src, src_mask, trg_mask, self_attn_mask)

        query_output = self.linear(q[:, 1:, :])
        content_output = self.linear(c[:, 1:, :])

        return query_output, content_output, q, c 
    
    def inference(self, trg, enc_src, src_mask, trg_mask=None):
        # inference step

        sz = trg.size() # (batch, seq_len_tgt)
        positions = self.pos_emb(trg) # (batch, seq_len_tgt, dim)

        q = trg.new_full(trg.size(), 0)  # (batch, seq_len_tgt)
        q = self.drop_out(self.tok_emb(q) + positions) # (batch, seq_len_tgt, dim)

        c = self.drop_out(self.tok_emb(trg) + positions) # (batch, seq_len_tgt, dim)
        self_attn_mask = self.buffered_future_mask(c)

        for layer in self.layers:
            q, c = layer(q, c, enc_src, src_mask, trg_mask, self_attn_mask)

        query_output = self.linear(q[:, -1, :])
        content_output = self.linear(c[:, -1, :])

        return query_output, content_output, q, c 

    def weight_forward(self, trg, enc_src, trg_mask, src_mask, use_weight_input, masked_tokens, query):
        
        positions = self.pos_emb(trg) # (batch, seq_len_tgt, dim)
        q = trg.new_full(trg.size(), 0)  # (batch, seq_len_tgt)
        q = self.drop_out(self.tok_emb(q) + positions) # (batch, seq_len_tgt+1, dim)
        c = self.drop_out(self.tok_emb(trg) + positions) # (batch, seq_len_tgt+1, dim)
        self_attn_mask = self.buffered_future_mask(c)

        if use_weight_input:
            embeddeding = self.tok_emb.weight
            softmax_input = F.softmax(query, dim=-1)
            embedded_item = self.drop_out(torch.einsum('bsv,ve->bse', softmax_input, embeddeding))
            c[:, 1:, :] = torch.where(masked_tokens.unsqueeze(-1), embedded_item, c[:, 1:, :])
        else:
            c[:, 1:, :] = torch.where(masked_tokens.unsqueeze(-1), query[:, 1:, :], c[:, 1:, :])

        for layer in self.layers:
            q, c = layer(q, c, enc_src, src_mask, trg_mask, self_attn_mask)

        query_output = self.linear(q[:, 1:, :])
        content_output = self.linear(c[:, 1:, :])

        return query_output, content_output, q, c 
    
    def inference_weight_forward(self, trg, enc_src, src_mask, use_weight_input, step, query, trg_mask=None):
        
        if step == 0:
            return self.inference(trg, enc_src, src_mask, trg_mask)
        else:
            positions = self.pos_emb(trg) # (batch, seq_len_tgt, dim)
            q = trg.new_full(trg.size(), 0)  # (batch, seq_len_tgt)
            q = self.drop_out(self.tok_emb(q) + positions) # (batch, seq_len_tgt+1, dim)
            c = self.drop_out(self.tok_emb(trg) + positions) # (batch, seq_len_tgt+1, dim)
            self_attn_mask = self.buffered_future_mask(c)

            if use_weight_input:
                embeddeding = self.tok_emb.weight
                softmax_input = F.softmax(query, dim=-1)
                embedded_item = self.drop_out(torch.einsum('bv,ve->be', softmax_input, embeddeding))
                c[:, -2, :] = embedded_item
            else:
                c[:, -2, :] = query[:, -1, :]

            for layer in self.layers:
                q, c = layer(q, c, enc_src, src_mask, trg_mask, self_attn_mask)

            query_output = self.linear(q[:, -1, :])
            content_output = self.linear(c[:, -1, :])

            return query_output, content_output, q, c 
    
class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)
    
    def forward(self, q, c, enc_src, src_mask, trg_mask, self_attn_mask):
        _q = q
        _c = c

        # 1. self att
        q, c = self.self_attention.two_stream_attention(
            query=q,
            content=c,
            key=c,
            value=c,
            key_padding_mask=trg_mask,
            attn_mask=self_attn_mask,
        )
        
        # 2. add and norm
        q = self.dropout1(q)
        q = self.norm1(q + _q)
        c = self.dropout1(c)
        c = self.norm1(c + _c)

        # 3. cross-att
        if enc_src is not None:
            # Support two modes:
            # - Standard: enc_src shape (batch, enc_len, dim)
            # - Memory-adaptor: enc_src shape (batch, mem_num, enc_num, dim), select enc chunk by decoder step
            if enc_src.dim() == 4:
                # Memory adaptor path
                bsz, mem_num, enc_num, dim = enc_src.shape

                # Q stream
                _q = q
                q_out_steps = []
                for step in range(q.shape[1]):
                    q_step = q[:, step, :].unsqueeze(1)
                    idx = mem_num - 1 if step >= mem_num else step
                    enc_step = enc_src[:, idx, :, :]
                    q_step_out = self.enc_dec_attention(q=q_step, k=enc_step, v=enc_step, mask=src_mask)
                    q_out_steps.append(q_step_out)
                q = torch.cat(q_out_steps, dim=1)
                q = self.dropout2(q)
                q = self.norm2(q + _q)

                # C stream
                _c = c
                c_out_steps = []
                for step in range(c.shape[1]):
                    c_step = c[:, step, :].unsqueeze(1)
                    idx = mem_num - 1 if step >= mem_num else step
                    enc_step = enc_src[:, idx, :, :]
                    c_step_out = self.enc_dec_attention(q=c_step, k=enc_step, v=enc_step, mask=src_mask)
                    c_out_steps.append(c_step_out)
                c = torch.cat(c_out_steps, dim=1)
                c = self.dropout2(c)
                c = self.norm2(c + _c)
            else:
                _q = q
                q = self.enc_dec_attention(q=q, k=enc_src, v=enc_src, mask=src_mask)
                q = self.dropout2(q)
                q = self.norm2(q + _q)

                _c = c
                c = self.enc_dec_attention(q=c, k=enc_src, v=enc_src, mask=src_mask)
                c = self.dropout2(c)
                c = self.norm2(c + _c)        

        # 4. feed forward network
        _q = q
        q = self.ffn(q)
        q = self.dropout3(q)
        q = self.norm3(q + _q)

        _c = c
        c = self.ffn(c)
        c = self.dropout3(c)
        c = self.norm3(c + _c)

        return q, c
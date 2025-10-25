import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x) 
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        return x.unsqueeze(0).contiguous()


class EncoderWithMemoryAdaptors(nn.Module):
    def __init__(self, enconder_fea_num, memory_adaptor_num, input_dim, hidden_dim, dropout_rate=0.5):
        super(EncoderWithMemoryAdaptors, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim * enconder_fea_num)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim * enconder_fea_num, hidden_dim * memory_adaptor_num * enconder_fea_num)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.enconder_fea_num = enconder_fea_num
        self.memory_adaptor_num = memory_adaptor_num
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        return x.view(x.shape[0], self.memory_adaptor_num, self.enconder_fea_num, self.hidden_dim).contiguous()
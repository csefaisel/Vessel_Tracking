import torch.nn.functional as F
import torch
import torch.nn as nn

class attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(attention, self).__init__()
        self.Wa = nn.Linear(hidden_size*2, attention_size, bias=False) # (2q x q)
        self.Ua = nn.Linear(hidden_size, attention_size, bias=False) # (q x q)
        self.Va = nn.Linear(attention_size, 1, bias=False) # (q x 1)

        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights with Xavier initialization
        nn.init.xavier_uniform_(self.Wa.weight)
        nn.init.xavier_uniform_(self.Ua.weight)
        nn.init.xavier_uniform_(self.Va.weight)
            
    def forward(self, query, values):
        # query: (batch_size, hidden_size)
        # values: (batch_size, seq_len, hidden_size)
        
        query = query.unsqueeze(1)  # (batch_size, 1, hidden_size)
        score = self.Va(torch.tanh(self.Wa(values) + self.Ua(query)))  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(score, dim=1)  # (batch_size, seq_len, 1)
        
        context_vector = torch.sum(attention_weights * values, dim=1)  # (batch_size, hidden_size)
        return context_vector

class vTrack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, out_seq_length):
        super(vTrack, self).__init__()
        self.num_layers = num_layers
        self.output_size = output_size
        self.out_seq_length = out_seq_length
        dec_input_size = hidden_size*2 + output_size

        self.enc = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attn = attention(hidden_size, hidden_size)
        self.dec = nn.LSTM(dec_input_size, hidden_size, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

        self.reset_parameters()

    def reset_parameters(self):
        
        # MLP weights initialized with the He method
        for lyr in [self.projection, self.fc]:
            for name, param in lyr.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif 'bias' in name:
                    nn.init.zeros_(param)

        for lyr in [self.enc, self.dec]:
            for name, param in lyr.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Forget gate bias initialization to 1
                    param.data[param.size(0) // 4 : param.size(0) // 2].fill_(1)          

    
    def forward(self, x, hidden=None):

        # BiLSTM-Encoder
        enc_out, enc_hidden = self.enc(x) # enc_out: (batch_size, seq_length, hidden_size*2)
                                          # enc_hidden: h: (num_layers*2, batch_size, hidden_size)
                                          #             c: (num_layers*2, batch_size, hidden_size)

        h_out_sequences = []
        for h in range(self.out_seq_length):

            # Decoder hidden_states
            if h == 0:
                # initializing decoder's hidden with encoder's hidden (forward)
                dh, dc = enc_hidden[0][:self.num_layers], enc_hidden[1][:self.num_layers]
                # implemented: tanh(Wk @ hl + bk)
                dh = torch.tanh(self.projection(dh))
                dec_hidden = (dh, dc)

            # Calculating context vector for each output sequence
            z_j = self.attn(dec_hidden[0][-1], enc_out) # (batch_size, hidden*2)
            
            # last input sequence x_l is the first input to decoder, otherwise decoder's previous output 
            y_j = x[:, -1, :self.output_size] if h == 0 else y_j
            # psi =    # Ψ (batch_size, journey_onehot)
            mew_t = torch.cat((y_j, z_j), dim=1) # if psi, then use torch.cat((y_j, z_j, psi), dim=1) instead
            
            # LSTM-Decoder
            _, dec_hidden = self.dec(mew_t.unsqueeze(dim=1), dec_hidden)
            # implemented: (Wy @ uj + by)
            y_j = self.fc(dec_hidden[0][-1]) 
            h_out_sequences.append(y_j)

        return torch.stack(h_out_sequences, dim=1) # (batch_size, out_seq_length, output_size)
        

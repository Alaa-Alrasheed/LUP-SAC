from torch import nn
from torch.nn import functional as F
import torch 

# embed_len = 50
# hidden_dim = 20
# n_layers=2

# class RNNClassifier(nn.Module):
#     def __init__(self):
#         super(RNNClassifier, self).__init__()
#         self.embedding_layer = nn.Embedding(num_embeddings=8240, embedding_dim=embed_len)
#         self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_dim, 4)

#     def forward(self, X_batch):
#         embeddings = self.embedding_layer(X_batch)
#         output, hidden = self.rnn(embeddings, torch.randn(n_layers, len(X_batch), hidden_dim))
#         return self.linear(output[:,-1])

# import torch
# from torch import nn

# embed_len = 50
# hidden_dim = 20
# n_layers = 3

# class RNNClassifier(nn.Module):
#     def __init__(self, vocab_size=8240, num_classes=4):
#         super(RNNClassifier, self).__init__()
#         self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_len)
#         self.lstm = nn.LSTM(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_dim, num_classes)

#     def forward(self, X_batch):
#         embeddings = self.embedding_layer(X_batch)
#         _, (hidden, _) = self.lstm(embeddings)
#         return self.linear(hidden[-1])


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init


class RNNClassifier(torch.nn.Module):
    def __init__(self, vocab_size=8240, embed_dim=40, hidden_dim=10, num_class=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.rnn = torch.nn.RNN(embed_dim,hidden_dim,batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, num_class)
        self.init_weights()
        
    def init_weights(self):
            initrange = 0.5
            
            # Initialize embedding layer weights
            self.embedding.weight.data.uniform_(-initrange, initrange)
            
            # Initialize RNN layer weights
            for name, param in self.rnn.named_parameters():
                if 'weight' in name:
                    init.uniform_(param.data, -initrange, initrange)
                elif 'bias' in name:
                    init.zeros_(param.data)
                    
            # Initialize linear layer weights and biases
            self.fc.weight.data.uniform_(-initrange, initrange)
            self.fc.bias.data.zero_()
    def forward(self, x):
                batch_size = x.size(1)
                x = self.embedding(x)
                x,h = self.rnn(x)
                return self.fc(x.mean(dim=1))   
# class RNNClassifier(nn.Module):
#     def __init__(self, vocab_size=8240, embed_dim=40, hidden_size=10, num_layers=2, num_class=4, dropout_prob=0.5):
#         super(RNNClassifier, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, bidirectional=True, batch_first=True)
#         self.dropout = nn.Dropout(dropout_prob)
#         self.fc = nn.Linear(hidden_size * 2, num_class)  # Multiply by 2 for bidirectional LSTM
        
#     def forward(self, text):
#         embedded = self.embedding(text)  # Apply word embeddings
#         lstm_out, _ = self.lstm(embedded)  # Pass through LSTM layers
#         lstm_out = self.dropout(lstm_out)  # Apply dropout
#         # Aggregate the bidirectional outputs using mean pooling
#         aggregated = torch.mean(lstm_out, dim=1)
#         logits = self.fc(aggregated)  # Apply fully connected layer
#         return logits   
    
# class RNNClassifier(nn.Module):
#     def __init__(self, vocab_size=8240, embed_dim=50, num_class=4):
#         super(RNNClassifier, self).__init__()
#         self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
#         self.fc = nn.Linear(embed_dim, num_class)
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.5
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.fc.weight.data.uniform_(-initrange, initrange)
#         self.fc.bias.data.zero_()

#     def forward(self, text):
#         embedded = self.embedding(text)
#         return self.fc(embedded)
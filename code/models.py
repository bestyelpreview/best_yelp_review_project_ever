"""!
@brief A wrapper model class which can be aptly utilized in order to
build any of the following models for polarity prediction:
LSTM
BLSTM
BLSTM with Attention on Top

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMPredictorWrapper(nn.Module):
    def __init__(self,
                 vocabulary_size=8000,
                 embedding_dimension=256,
                 dropout_rate=0.0,
                 num_layers=1,
                 num_hidden_units=512,
                 LSTM_type='LSTM',
                 return_attention_weights=False):
        """
        :param vocabulary_size: The size of vocabulary used in the
        embedding layer.
        :param embedding_dimension: The size of the output vector for
        each token.
        :param dropout_rate:
        :param num_layers: Number of hidden layers in the RNN.
        :param num_hidden_units: Number of hidden units for each layer
        in the selected model.
        :param LSTM_type: The type of model which is going to be used
        for prediction.
        Options:
            1) LSTM
            2) A bidirectional LSTM (BLSTM)
            3) BLSTM with Attention on top (BLSTM-Att)
        :param return_atention_weights: Whether you need the
        attention map to be returned.
        """
        super(LSTMPredictorWrapper, self).__init__()

        self.V = vocabulary_size
        self.E = embedding_dimension
        self.D = dropout_rate
        self.L = num_layers
        self.H = num_hidden_units
        self.model_type = LSTM_type
        self.return_attention_weights = return_attention_weights

        self.apply_attention = False
        if LSTM_type == 'LSTM':
            self.bidirectional = False
        elif LSTM_type == 'BLSTM':
            self.bidirectional = True
        elif LSTM_type == 'BLSTM-Att':
            self.bidirectional = True
            self.apply_attention = True
        else:
            raise NotImplementedError("LSTM of type: {} is not "
                "implemented in this wrapper!".format(LSTM_type))
        self.n_directions = 2**int(self.bidirectional)

        self.embedding = nn.Embedding(self.V, self.E)
        self.input_dropout = nn.Dropout(self.D)
        self.lstm = nn.LSTM(input_size=self.E,
                            hidden_size=self.H,
                            num_layers=self.L,
                            dropout=self.D,
                            batch_first=True,
                            bidirectional=self.bidirectional)
        self.out = nn.Linear(self.H * self.n_directions, 1)

    def attention(self, lstm_output, final_state):
        # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        hidden = final_state.view(-1, self.H * self.n_directions, 1)
        # attn_weights : [batch_size, n_step]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2),
                            soft_attn_weights.unsqueeze(2)).squeeze(2)
        # context : [batch_size, n_hidden * num_directions(=2)]

        if self.return_attention_weights:
            return context, soft_attn_weights.detach().cpu().numpy()
        else:
            return context, None

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = self.input_dropout(x)
        # input : [batch_size, len_seq, embedding_dim]

        if lengths is not None:
            x_pack = pack_padded_sequence(x, lengths, batch_first=True)
            lstm_output, (final_hidden_state, final_cell_state) = \
                self.lstm(x_pack)
            # output : [batch_size, len_seq, n_hidden]
            lstm_output, _ = pad_packed_sequence(lstm_output,
                                                 batch_first=True)
        else:
            lstm_output, (final_hidden_state, final_cell_state) = \
                self.lstm(x)

        if self.apply_attention:
            attn_output, attention = self.attention(
                lstm_output,
                final_hidden_state[-self.n_directions:].permute(
                    1, 0, 2).reshape(-1, self.H * self.n_directions))
            prediction = self.out(attn_output.squeeze())
            # model : [batch_size, num_classes], attention : [batch_size, n_step]
        else:
            attention = None
            prediction = self.out(
                final_hidden_state[-self.n_directions:].permute(
                    1, 0, 2).reshape(-1, self.H * self.n_directions))

        if self.return_attention_weights:
            return prediction, attention
        return prediction

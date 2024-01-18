import torch
from torch import nn

class E2MLP(nn.Module):
    def __init__(self, num_nodes, time_of_day_size, day_of_week_size, input_dim, input_len, output_len, dropout,
                 num_layer, num_block, timeseries_embed_dim, sid_embed_dim, tid_embed_dim, hidden_dim):
        super().__init__()
        # attributes
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.dropout = dropout
        self.num_block = num_block
        self.num_layer = num_layer
        self.timeseries_embed_dim = timeseries_embed_dim
        self.node_dim = sid_embed_dim
        self.tid_embed_dim = tid_embed_dim
        self.temp_dim_diw = tid_embed_dim
        self.hidden_dim = hidden_dim
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

        self.if_time_in_day = True
        self.if_day_in_week = True
        self.if_spatial = True

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.tid_embed_dim))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.tid_embed_dim))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.timeseries_embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.mix_layers = nn.ModuleList()
        emb_dim = self.if_spatial * self.node_dim + self.if_time_in_day*2*self.tid_embed_dim
        self.mix_layers.append(nn.Conv2d(self.timeseries_embed_dim+emb_dim,
                                         self.hidden_dim, kernel_size=(1, 1), bias=True))
        for i in range(1, self.num_block):
            self.mix_layers.append(nn.Conv2d(self.hidden_dim+emb_dim,
                                             self.hidden_dim, kernel_size=(1, 1), bias=True))
        self.encoders = nn.ModuleList()
        self.forecast_layers = nn.ModuleList()
        self.backcast_layers = nn.ModuleList()
        for i in range(0, self.num_block):
            self.encoders.append(TimeBlock(hidden_dim=self.hidden_dim, dropout=self.dropout, num_layer=self.num_layer))
            self.forecast_layers.append(nn.Conv2d(in_channels=hidden_dim, out_channels=self.output_len,
                                                  kernel_size=(1, 1), bias=True))
            self.backcast_layers.append(nn.Conv2d(in_channels=hidden_dim, out_channels=self.hidden_dim,
                                                  kernel_size=(1, 1), bias=True))


    def forward(self, history_data: torch.Tensor) -> torch.Tensor:

        # prepare data
        input_data = history_data[..., range(self.input_dim)]


        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        hidden = self.time_series_emb_layer(input_data)

        if self.if_spatial:
            node_emb = self.node_emb.unsqueeze(0).expand(
                    batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
        else:
            node_emb = None
        if self.if_time_in_day:
            tem_emb_day = time_in_day_emb.transpose(1, 2).unsqueeze(-1)
            temp_emb_week = day_in_week_emb.transpose(1, 2).unsqueeze(-1)
        else:
            tem_emb_day = None
            temp_emb_week = None
        # encoding
        prediction = torch.zeros(batch_size, self.output_len, num_nodes, 1).cuda()
        for i in range(0, self.num_block):
            if self.if_spatial:
                hidden = torch.cat((hidden, node_emb), dim=1)
            if self.if_time_in_day:
                hidden = torch.cat((hidden, tem_emb_day, temp_emb_week), dim=1)
            hidden = self.mix_layers[i](hidden)
            hidden = self.encoders[i](hidden)
            prediction += self.forecast_layers[i](hidden)

        return prediction

class TimeMlp(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.15):
        super(TimeMlp, self).__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm2d(num_features=output_dim),
            nn.Conv2d(in_channels=input_dim,  out_channels=output_dim, kernel_size=(1, 1), bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return x + self.mlp(x)
class TimeBlock(nn.Module):
    def __init__(self, hidden_dim, dropout, num_layer):
        super(TimeBlock, self).__init__()
        self.time_layers = nn.Sequential(
            *[TimeMlp(input_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout) for _ in range(num_layer)]
        )

    def forward(self, x):
        x = self.time_layers(x)
        return x

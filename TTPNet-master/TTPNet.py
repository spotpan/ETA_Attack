import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np

class Road(nn.Module):
    def __init__(self):
        super(Road, self).__init__()
        self.build()

    def build(self):
        self.embedding = nn.Embedding(128*128, 32)
        emb_vectors = np.load('Config/embedding_128.npy')
        self.embedding.weight.data.copy_(torch.from_numpy(emb_vectors))
        self.process_coords = nn.Linear(2+32, 32)
        
#        for module in self.modules():
#            if type(module) is not nn.Embedding:
#                continue
#            nn.init.uniform_(module.state_dict()['weight'], a=-1, b=1)

    def forward(self, traj):
        # Convert to tensor if necessary
        if not isinstance(traj['lngs'], torch.Tensor):
            traj['lngs'] = torch.stack([torch.tensor(lng, dtype=torch.float32) for lng in traj['lngs']])

        if not isinstance(traj['lats'], torch.Tensor):
            traj['lats'] = torch.stack([torch.tensor(lat, dtype=torch.float32) for lat in traj['lats']])

        lngs = torch.unsqueeze(traj['lngs'], dim=2)
        lats = torch.unsqueeze(traj['lats'], dim=2)
        grid_ids = torch.unsqueeze(traj['grid_id'].long(), dim=2)
        grids = torch.squeeze(self.embedding(grid_ids))
        
        locs = torch.cat([lngs, lats, grids], dim=2)
        locs = self.process_coords(locs)
        locs = F.tanh(locs)
        
        return locs






class ShortSpeed(nn.Module):
    def __init__(self):
        super(ShortSpeed, self).__init__()
        self.build()
    
    def build(self):
#        self.process_shortspeeds = nn.Linear(48, 16)
        self.short_kernel_size = 2
        self.short_cnn = nn.Conv1d(3, 4, kernel_size = self.short_kernel_size, stride = 1)
        self.short_rnn = nn.RNN(
            input_size = 4, \
            hidden_size = 16, \
            num_layers = 1, \
            batch_first = True
        )
        
#        nn.init.uniform_(self.short_rnn.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)
    
    def forward(self, traj):
        # short-term travel speed features
        n_batchs = traj['speeds_0'].size()[0]
        speeds_forward = traj['speeds_0'].reshape(-1, 4)
        speeds_adjacent1 = traj['speeds_1'].reshape(-1, 4)
        speeds_adjacent2 = traj['speeds_2'].reshape(-1, 4)
        grid_len = traj['grid_len'].reshape(-1, 1)
        
        speeds_forward = torch.unsqueeze(speeds_forward, dim =2)
        speeds_adjacent1 = torch.unsqueeze(speeds_adjacent1, dim = 2)
        speeds_adjacent2 = torch.unsqueeze(speeds_adjacent2, dim = 2)

        grid_len = torch.unsqueeze(grid_len, dim = 2)
        grid_len_short = grid_len.expand(speeds_forward.size()[:2] + (grid_len.size()[-1], ))
        
        times_forward = speeds_forward.clone()
        times_forward[times_forward==0] = 0.2
        times_forward = grid_len_short / times_forward * 3600
        times_adjacent1 = speeds_adjacent1.clone()
        times_adjacent1[times_adjacent1==0] = 0.2
        times_adjacent1 = grid_len_short / times_adjacent1 * 3600
        times_adjacent2 = speeds_adjacent2.clone()
        times_adjacent2[times_adjacent2==0] = 0.2
        times_adjacent2 = grid_len_short / times_adjacent2 * 3600
        
        speeds_forward = utils.normalize(speeds_forward, 'speeds_0')
        speeds_adjacent1 = utils.normalize(speeds_adjacent1, 'speeds_1')
        speeds_adjacent2 = utils.normalize(speeds_adjacent2, 'speeds_2')
        grid_len_short = utils.normalize(grid_len_short, 'grid_len')
        times_forward = utils.normalize(times_forward, 'time_gap')
        times_adjacent1 = utils.normalize(times_adjacent1, 'time_gap')
        times_adjacent2 = utils.normalize(times_adjacent2, 'time_gap')
        
        inputs_0 = torch.cat([speeds_forward, grid_len_short, times_forward], dim = 2)
        inputs_1 = torch.cat([speeds_adjacent1, grid_len_short, times_adjacent1], dim = 2)
        inputs_2 = torch.cat([speeds_adjacent2, grid_len_short, times_adjacent2], dim = 2)
        
        outputs_0 = F.tanh(self.short_cnn(inputs_0.permute(0, 2, 1)))
        outputs_0 = outputs_0.permute(0, 2, 1)
        outputs_1 = F.tanh(self.short_cnn(inputs_1.permute(0, 2, 1)))
        outputs_1 = outputs_1.permute(0, 2, 1)
        outputs_2 = F.tanh(self.short_cnn(inputs_2.permute(0, 2, 1)))
        outputs_2 = outputs_2.permute(0, 2, 1)

        outputs_0, h_n = self.short_rnn(outputs_0)
        outputs_1, h_n = self.short_rnn(outputs_1)
        outputs_2, h_n = self.short_rnn(outputs_2)

        outputs_0 = outputs_0.reshape(n_batchs, -1, 4-self.short_kernel_size+1, 16)
        outputs_1 = outputs_1.reshape(n_batchs, -1, 4-self.short_kernel_size+1, 16)
        outputs_2 = outputs_2.reshape(n_batchs, -1, 4-self.short_kernel_size+1, 16)
        
        V_short = torch.cat([outputs_0[:, :, -1], outputs_1[:, :, -1], outputs_2[:, :, -1]], dim = 2)
#        V_short = self.process_shortspeeds(V_short)
#        V_short = F.tanh(V_short)
        
        return V_short


class LongSpeed(nn.Module):
    def __init__(self):
        super(LongSpeed, self).__init__()
        self.build()
    
    def build(self):
#        self.process_longspeeds = nn.Linear(16, 16)
        self.long_kernel_size = 3
        self.long_cnn = nn.Conv1d(3, 4, kernel_size = self.long_kernel_size, stride = 1)
        self.long_rnn = nn.RNN(
            input_size = 4, \
            hidden_size = 16, \
            num_layers = 1, \
            batch_first = True
        )

#        nn.init.uniform_(self.long_rnn.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)
    
    def forward(self, traj):
        # long-term travel speed features
        n_batchs = traj['speeds_long'].size()[0]
        speeds_history = traj['speeds_long'].reshape(-1, 7)
        grid_len = traj['grid_len'].reshape(-1, 1)
        
        speeds_history = torch.unsqueeze(speeds_history, dim = 2)

        grid_len = torch.unsqueeze(grid_len, dim = 2)
        grid_len_long = grid_len.expand(speeds_history.size()[:2] + (grid_len.size()[-1], ))
        
        times_history = speeds_history.clone()
        times_history[times_history==0] = 0.2
        times_history = grid_len_long / times_history * 3600
        
        speeds_history = utils.normalize(speeds_history, 'speeds_long')
        grid_len_long = utils.normalize(grid_len_long, 'grid_len')
        times_history = utils.normalize(times_history, 'time_gap')
        
        inputs_3 = torch.cat([speeds_history, grid_len_long, times_history], dim = 2)
        outputs_3 = self.long_cnn(inputs_3.permute(0, 2, 1))
        outputs_3 = outputs_3.permute(0, 2, 1)
        outputs_3, h_n = self.long_rnn(outputs_3)
        outputs_3 = outputs_3.reshape(n_batchs, -1, 7-self.long_kernel_size+1, 16)
        
        V_long = outputs_3[:, :, -1]
#        V_long = self.process_longspeeds(V_long)
#        V_long = F.tanh(V_long)
        
        return V_long

class SpeedLSTM(nn.Module):
    def __init__(self):
        super(SpeedLSTM, self).__init__()
        self.shortspeed_net = ShortSpeed()
        self.longspeed_net = LongSpeed()
        self.process_speeds = nn.Linear(64, 32)
        self.speed_lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )

    def forward(self, attr, traj):
        shortspeeds_t = self.shortspeed_net(traj)
        longspeeds_t = self.longspeed_net(traj)
        whole_t = torch.cat([shortspeeds_t, longspeeds_t], dim=2)
        whole_t = self.process_speeds(whole_t)
        whole_t = torch.tanh(whole_t)

        # Ensure lens is accurate
        lens = copy.deepcopy(traj['lens'])
        if isinstance(lens, map):
            lens = list(lens)
        assert len(lens) == whole_t.size(0), "Mismatch between lens and batch size"
        for idx, l in enumerate(lens):
            assert l <= whole_t.size(1), f"Length {l} exceeds max sequence length {whole_t.size(1)} at index {idx}"

        # Pack and unpack the sequence
        packed_inputs = nn.utils.rnn.pack_padded_sequence(whole_t, lens, batch_first=True, enforce_sorted=False)
        packed_hiddens, (h_n, c_n) = self.speed_lstm(packed_inputs)
        speeds_hiddens, unpacked_lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        # Debug output
        # print(f"Packed inputs shape: {packed_hiddens.data.size()}")
        # print(f"Speeds hiddens shape: {speeds_hiddens.size()}, Unpacked lens: {unpacked_lens}")

        return speeds_hiddens

"""
class SpeedLSTM(nn.Module):

    def __init__(self, ):
        super(SpeedLSTM, self).__init__()
#        self.attr_net = Attr()
        self.shortspeed_net = ShortSpeed()
        self.longspeed_net = LongSpeed()
        self.process_speeds = nn.Linear(64, 32)
#        self.process_speeds_hiddens = nn.Linear(64, 32)
        self.speed_lstm = nn.LSTM(
            input_size = 32, \
            hidden_size = 32, \
            num_layers = 1, \
            batch_first = True, \
            bidirectional = False, \
            dropout = 0
        )

#        nn.init.uniform_(self.speed_lstm.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)

    def forward(self, attr, traj):
        shortspeeds_t = self.shortspeed_net(traj)
        longspeeds_t = self.longspeed_net(traj)
#        attr_t = self.attr_net(attr)
#        attr_t = torch.unsqueeze(attr_t, dim = 1)
#        expand_attr_t = attr_t.expand(speeds_t.size()[:2] + (attr_t.size()[-1], ))
#        whole_t = torch.cat([expand_attr_t, speeds_t], dim = 2)
        whole_t = torch.cat([shortspeeds_t, longspeeds_t], dim = 2)
        whole_t = self.process_speeds(whole_t)
        whole_t = F.tanh(whole_t)
        
        lens = copy.deepcopy(traj['lens'])
        lens = list(map(lambda x: x, lens))
        # Fetch and print the original lens
        #lens = traj['lens']
        print(f"Original lens: {lens}")
        
        packed_inputs = nn.utils.rnn.pack_padded_sequence(whole_t, lens, batch_first = True, enforce_sorted=False)
        packed_hiddens, (h_n, c_n) = self.speed_lstm(packed_inputs)
        speeds_hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first = True)

        print(f"Speeds hiddens shape: {speeds_hiddens.shape}, Lens: {lens}")
        
#        speeds_hiddens = self.process_speeds_hiddens(speeds_hiddens)
#        speeds_hiddens = F.tanh(speeds_hiddens)
        
        return speeds_hiddens
"""


class RoadLSTM(nn.Module):

    def __init__(self, ):
        super(RoadLSTM, self).__init__()
#        self.attr_net = Attr()
        self.Road_net = Road()
#        self.process_Roads_hiddens = nn.Linear(64, 32)
        self.Road_lstm = nn.LSTM(
            input_size = 32, \
            hidden_size = 32, \
            num_layers = 1, \
            batch_first = True, \
            bidirectional = False, \
            dropout = 0
        )

#        nn.init.uniform_(self.Road_lstm.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)

    def forward(self, attr, traj):
        Roads_t = self.Road_net(traj)
#        attr_t = self.attr_net(attr)
#        attr_t = torch.unsqueeze(attr_t, dim = 1)
#        expand_attr_t = attr_t.expand(Roads_t.size()[:2] + (attr_t.size()[-1], ))
#        whole_t = torch.cat([expand_attr_t, Roads_t], dim = 2)
        whole_t = Roads_t
        
        lens = copy.deepcopy(traj['lens'])
        lens = list(map(lambda x: x, lens))
        
        packed_inputs = nn.utils.rnn.pack_padded_sequence(whole_t, lens, batch_first = True)
        packed_hiddens, (h_n, c_n) = self.Road_lstm(packed_inputs)
        Roads_hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first = True)
        
#        Roads_hiddens = self.process_Roads_hiddens(Roads_hiddens)
#        Roads_hiddens = F.tanh(Roads_hiddens)
        
        return Roads_hiddens




class Attr(nn.Module):
    embed_dims = [('driverID', 13000, 8), ('weekID', 7, 3), ('timeID', 96, 8)]

    def __init__(self):
        super(Attr, self).__init__()
        # whether to add the two ends of the path into Attribute Component
        self.build()

    def build(self):
        for name, dim_in, dim_out in Attr.embed_dims:
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))
        
#        for module in self.modules():
#            if type(module) is not nn.Embedding:
#                continue
#            nn.init.uniform_(module.state_dict()['weight'], a=-1, b=1)

    def out_size(self):
        sz = 0
        for name, dim_in, dim_out in Attr.embed_dims:
            sz += dim_out
        
        return sz + 2

    def forward(self, attr):
        em_list = []
        for name, dim_in, dim_out in Attr.embed_dims:
            embed = getattr(self, name + '_em')
            attr_t = attr[name].view(-1, 1)

            attr_t = torch.squeeze(embed(attr_t))

            em_list.append(attr_t)

        dist = attr['dist']
        em_list.append(dist.view(-1, 1))
        em_list.append(attr['dateID'].float().view(-1, 1))

        return torch.cat(em_list, dim = 1)


class PredictionBiLSTM(nn.Module):

    def __init__(self):
        super(PredictionBiLSTM, self).__init__()
        self.build()
    
    def build(self):
        self.attr_net = Attr()
        self.speed_lstm = SpeedLSTM()
        self.road_lstm = RoadLSTM()
        self.bi_lstm = nn.LSTM(
            input_size = self.attr_net.out_size() + 64, \
            hidden_size = 64, \
            num_layers = 2, \
            batch_first = True, \
            bidirectional = True, \
            dropout = 0.25
        )
        
        self.lnhiddens = nn.LayerNorm(self.attr_net.out_size() + 64, elementwise_affine=True)
#        nn.init.uniform_(self.bi_lstm.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)
        
    def forward(self, attr, traj):
        speeds_t = self.speed_lstm(attr, traj)
        roads_t = self.road_lstm(attr, traj)
        
        attr_t = self.attr_net(attr)
        attr_t = torch.unsqueeze(attr_t, dim = 1)
        expand_attr_t = attr_t.expand(roads_t.size()[:2] + (attr_t.size()[-1], ))

        hiddens = torch.cat([expand_attr_t, speeds_t, roads_t], dim = 2)
        hiddens = self.lnhiddens(hiddens)
        lens = copy.deepcopy(traj['lens'])
        lens = list(map(lambda x: x, lens))
        
        packed_inputs = nn.utils.rnn.pack_padded_sequence(hiddens, lens, batch_first = True)
        packed_hiddens, (h_n, c_n) = self.bi_lstm(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first = True)

        return hiddens
        

class TTPNet(nn.Module):
    def __init__(self, ):
        super(TTPNet, self).__init__()

        self.build()
#        self.init_weight()

    def build(self):
        self.bi_lstm = PredictionBiLSTM()
        
        self.input2hid = nn.Linear(128, 128)
        self.hid2hid = nn.Linear(128, 64)
        self.hid2out = nn.Linear(64, 1)
        
    def forward(self, attr, traj):
        hiddens = self.bi_lstm(attr, traj)
        n = hiddens.size()[1]
        h_f = []
        
        for i in range(2, n):
            h_f_temp = torch.sum(hiddens[:, :i], dim = 1)
            h_f.append(h_f_temp)
            
        h_f.append(torch.sum(hiddens, dim = 1))
        h_f = torch.stack(h_f).permute(1, 0, 2)
        
        T_f_hat = self.input2hid(h_f)
        T_f_hat = F.relu(T_f_hat)
        T_f_hat = self.hid2hid(T_f_hat)
        T_f_hat = F.relu(T_f_hat)
        T_f_hat = self.hid2out(T_f_hat)

        return T_f_hat

    def dual_loss(self, T_f_hat, traj, mean, std):
        T_f_hat = T_f_hat * std + mean
        
        T_f = torch.unsqueeze(traj['T_f'], dim = 2)
        M_f = torch.unsqueeze(traj['M_f'], dim = 1)
        
        loss_f = torch.bmm(M_f, torch.pow((T_f_hat-T_f)/T_f, 2)) / torch.bmm(M_f, M_f.permute(0, 2, 1))
        loss_f = torch.pow(loss_f, 1/2)
        
        return {'pred': T_f_hat[:, -1]}, loss_f.mean()
    
    def MAPE_loss(self, pred, label, mean, std):
        label = label.view(-1, 1)
        label = label * std + mean
        
        loss = torch.abs(pred - label) / label
        
        return {'label': label, 'pred': pred}, loss.mean()
    
#    def init_weight(self):
#        for name, param in self.named_parameters():
#            if name.find('.ln') == -1:
#                print(name)
#                if name.find('.bias') != -1:
#                    param.data.fill_(0)
#                elif name.find('.weight') != -1:
#                    nn.init.xavier_uniform_(param.data)
    
    def eval_on_batch(self, attr, traj, config):
        T_f_hat = self(attr, traj)
        if self.training:
            pred_dict, loss = self.dual_loss(T_f_hat, traj, config['time_gap_mean'], config['time_gap_std'])
            return pred_dict, loss
        else:
#            pred_dict, loss = self.dual_loss(T_f_hat, traj, config['time_gap_mean'], config['time_gap_std'])
#            MAPE_dict, MAPE_loss = self.MAPE_loss(pred_dict['pred'], attr['time'], config['time_mean'], config['time_std'])
            pred = T_f_hat * config['time_gap_std'] + config['time_gap_mean']
            MAPE_dict, MAPE_loss = self.MAPE_loss(pred[:, -1], attr['time'], config['time_mean'], config['time_std'])
            return MAPE_dict, MAPE_loss

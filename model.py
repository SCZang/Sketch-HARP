import torch
import numpy as np
import torch.nn.functional as F

class DropPath(torch.nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

class SpatialGatingUnit(torch.nn.Module):
    def __init__(self, d_ffn, seq_len):
        super(SpatialGatingUnit, self).__init__()
        self.norm = torch.nn.LayerNorm(d_ffn // 2)
        self.spatial_proj = torch.nn.Conv1d(seq_len, seq_len, 1)
        # self.attn = Attention(d_ffn, d_ffn//2, 64)

    def forward(self, x):
        u, v = torch.chunk(x, 2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)  # + self.attn(x)
        out = u * v
        return out

class GMLPblock(torch.nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, dpr=0.0):
        super(GMLPblock, self).__init__()
        self.norm = torch.nn.LayerNorm(d_model)
        self.channel_proj1 = torch.nn.Linear(d_model, d_ffn)
        self.channel_proj2 = torch.nn.Linear(d_ffn // 2, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

        self.droppath = DropPath(dpr) if dpr > 0.0 else torch.nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = torch.nn.functional.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        return shortcut + self.droppath(x)

class SeqEncoder(torch.nn.Module):
    '''lstm-1 for learning stroke embeddings'''
    def __init__(self, args):
        super(SeqEncoder, self).__init__()

        self.zdim = args.zdim
        self.enc_rnn_size_1 = args.enc_rnn_size_1
        self.stroke_emd_size = args.stroke_emd_size
        self.dropout = 0.1

        self.lstm_1 = torch.nn.LSTM(
            5,
            self.enc_rnn_size_1,
            bidirectional=True,
            # dropout=self.dropout,
            batch_first=True)

        self.stroke_embedding_predictor = torch.nn.Linear(2 * self.enc_rnn_size_1, self.stroke_emd_size)

        self.pos2emb = torch.nn.Linear(2, self.stroke_emd_size)

        self.layers = torch.nn.ModuleList([GMLPblock(self.stroke_emd_size, self.stroke_emd_size * 4, args.max_stroke_num, dpr=0.1) for i in range(2)])

    def forward(self, padded_strokes, start_positions, stroke_lens, mask, hidden_and_cell_1=None):
        bs, max_stroke_num, max_stroke_len = padded_strokes.shape[0:3]

        padded_strokes = padded_strokes.contiguous().reshape(bs * max_stroke_num, max_stroke_len, 5)
        stroke_lens = stroke_lens.contiguous().reshape(bs * max_stroke_num)

        if hidden_and_cell_1 is None:
            hidden_1 = torch.zeros(2, bs * max_stroke_num, self.enc_rnn_size_1).cuda()
            cell_1 = torch.zeros(2, bs * max_stroke_num, self.enc_rnn_size_1).cuda()
            hidden_and_cell_1 = (hidden_1, cell_1)

        packed_strokes = torch.nn.utils.rnn.pack_padded_sequence(padded_strokes, stroke_lens, batch_first=True, enforce_sorted=False)
        packed_outputs_1, hidden_and_cell_1 = self.lstm_1(packed_strokes, hidden_and_cell_1)
        # outputs_1, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs_1, batch_first=True, total_length=max_stroke_len)

        # outputs_1_fw, outputs_1_bw = torch.split(outputs_1, [self.enc_rnn_size_1, self.enc_rnn_size_1], dim=2)
        hidden_1, cell_1 = hidden_and_cell_1  # 2, N * max_stroke_num, d
        hidden_1_fw, hidden_1_bw = torch.split(hidden_1, 1, dim=0)
        last_h_1 = torch.cat([hidden_1_fw.squeeze(0), hidden_1_bw.squeeze(0)], dim=1)

        stroke_embeddings = self.stroke_embedding_predictor(last_h_1)
        stroke_embeddings = stroke_embeddings.contiguous().reshape(bs, max_stroke_num, self.stroke_emd_size)
        pe = self.pos2emb(start_positions)

        relationship_embeddings = (stroke_embeddings + pe) * mask.unsqueeze(dim=2).to(pe.device)
        for layer in self.layers:
            relationship_embeddings = layer(relationship_embeddings)

        return relationship_embeddings + stroke_embeddings, pe

class StrokeEncoder(torch.nn.Module):
    def __init__(self, args):
        super(StrokeEncoder, self).__init__()

        self.zdim = args.zdim
        self.enc_rnn_size_2 = args.enc_rnn_size_2
        self.stroke_emd_size = args.stroke_emd_size
        self.dropout = 0.1

        self.lstm_2 = torch.nn.LSTM(
            self.stroke_emd_size,
            self.enc_rnn_size_2,
            bidirectional=True,
            # dropout=self.dropout,
            batch_first=True)

        self.z_predictor = torch.nn.Linear(2 * self.enc_rnn_size_2, self.zdim)

    def forward(self, stroke_embeddings, rel_pe, stroke_nums, hidden_and_cell_2=None):
        bs = stroke_embeddings.shape[0]

        if hidden_and_cell_2 is None:
            hidden_2 = torch.zeros(2, bs, self.enc_rnn_size_2).cuda()
            cell_2 = torch.zeros(2, bs, self.enc_rnn_size_2).cuda()
            hidden_and_cell_2 = (hidden_2, cell_2)

        packed_seqs = torch.nn.utils.rnn.pack_padded_sequence(stroke_embeddings + rel_pe, stroke_nums, batch_first=True, enforce_sorted=False)
        packed_outputs_2, hidden_and_cell_2 = self.lstm_2(packed_seqs, hidden_and_cell_2)
        # outputs_2, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs_2, batch_first=True, total_length=max_stroke_num)

        hidden_2, cell_2 = hidden_and_cell_2  # 2, N, hs
        hidden_2_fw, hidden_2_bw = torch.split(hidden_2, 1, dim=0)
        last_h_2 = torch.cat([hidden_2_fw.squeeze(0), hidden_2_bw.squeeze(0)], dim=1)

        z = self.z_predictor(last_h_2)

        return z

class StrokeDecoder(torch.nn.Module):
    '''lstm-1 to predict stroke and position embeddings'''
    def __init__(self, args):
        super(StrokeDecoder, self).__init__()
        self.stroke_emd_size = args.stroke_emd_size
        self.dec_rnn_size_1 = args.dec_rnn_size_1
        self.num_layers = 1
        self.zdim = args.zdim
        self.num_gaussians = 20
        self.dropout = 0.1

        # Maps the latent vector to an initial cell/hidden vector
        self.hidden_cell_1_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.zdim, 2 * self.dec_rnn_size_1),
            torch.nn.Tanh()
        )

        self.lstm_1 = torch.nn.LSTM(
            self.stroke_emd_size + self.zdim,
            self.dec_rnn_size_1,
            num_layers=self.num_layers,
            # dropout=self.dropout,
            batch_first=True)

        self.parameters_predictor = torch.nn.Linear(self.dec_rnn_size_1, self.stroke_emd_size + 2)

    def get_params(self, output):
        pred_stroke_emb = output[:, :, 0:self.stroke_emd_size]

        stopper_logits = output[:, :, self.stroke_emd_size:]
        pred_stopper = torch.nn.functional.softmax(stopper_logits, dim=-1)
        return pred_stroke_emb, stopper_logits, pred_stopper

    def forward(self, z, padded_stroke_emb, padded_pos_emb, hidden_and_cell_1=None):
        bs, max_stroke_num = padded_stroke_emb.shape[0:2]

        expanded_z = z.unsqueeze(1).repeat(1, max_stroke_num, 1)
        inputs_1 = torch.cat([padded_stroke_emb + padded_pos_emb, expanded_z], dim=2)

        if hidden_and_cell_1 is None:
            hidden_and_cell_1 = self.hidden_cell_1_predictor(z)
            hidden_1 = hidden_and_cell_1[:, :self.dec_rnn_size_1]
            hidden_1 = hidden_1.unsqueeze(0).contiguous()
            cell_1 = hidden_and_cell_1[:, self.dec_rnn_size_1:]
            cell_1 = cell_1.unsqueeze(0).contiguous()
            hidden_and_cell_1 = (hidden_1, cell_1)

        outputs_1, hidden_and_cell_1 = self.lstm_1(inputs_1, hidden_and_cell_1)
        params = self.parameters_predictor(outputs_1)
        pred_stroke_emb, stopper_logits, pred_stopper = self.get_params(params)

        return pred_stroke_emb, stopper_logits, pred_stopper, hidden_and_cell_1

class PosDecoder(torch.nn.Module):
    '''lstm-1 to predict stroke and position embeddings'''
    def __init__(self, args):
        super(PosDecoder, self).__init__()
        self.stroke_emd_size = args.stroke_emd_size
        self.dec_rnn_size_3 = args.dec_rnn_size_3
        self.num_layers = 1
        self.zdim = args.zdim
        self.dropout = 0.1

        # Maps the latent vector to an initial cell/hidden vector
        self.hidden_cell_3_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.zdim, 2 * self.dec_rnn_size_3),
            torch.nn.Tanh()
        )

        self.lstm_3 = torch.nn.LSTM(
            2 * self.stroke_emd_size + self.zdim,
            self.dec_rnn_size_3,
            num_layers=self.num_layers,
            # dropout=self.dropout,
            batch_first=True)

        self.parameters_predictor = torch.nn.Linear(self.dec_rnn_size_3, 5)

    def get_params(self, pred_pos_emb):
        mu1, mu2, sigma1, sigma2, corr = torch.split(pred_pos_emb, 1, dim=2)
        sigma1 = sigma1.exp()
        sigma2 = sigma2.exp()
        corr = torch.nn.Tanh()(corr)
        return mu1, mu2, sigma1, sigma2, corr

    def forward(self, z, padded_stroke_emb, padded_pos_emb, hidden_and_cell_3=None):
        bs, max_stroke_num = padded_pos_emb.shape[0:2]

        expanded_z = z.unsqueeze(1).repeat(1, max_stroke_num, 1)
        inputs_3 = torch.cat([padded_stroke_emb[:, :-1] + padded_pos_emb, padded_stroke_emb[:, 1:], expanded_z], dim=2)

        if hidden_and_cell_3 is None:
            hidden_and_cell_3 = self.hidden_cell_3_predictor(z)
            hidden_3 = hidden_and_cell_3[:, :self.dec_rnn_size_3]
            hidden_3 = hidden_3.unsqueeze(0).contiguous()
            cell_3 = hidden_and_cell_3[:, self.dec_rnn_size_3:]
            cell_3 = cell_3.unsqueeze(0).contiguous()
            hidden_and_cell_3 = (hidden_3, cell_3)

        outputs_3, hidden_and_cell_3 = self.lstm_3(inputs_3, hidden_and_cell_3)
        params = self.parameters_predictor(outputs_3)
        mu1, mu2, sigma1, sigma2, corr = self.get_params(params)

        return mu1, mu2, sigma1, sigma2, corr, hidden_and_cell_3

class SeqDecoder(torch.nn.Module):
    '''lstm-2 to predict drawing actions'''
    def __init__(self, args):
        super(SeqDecoder, self).__init__()
        self.stroke_emd_size = args.stroke_emd_size
        self.dec_rnn_size_2 = args.dec_rnn_size_2
        self.num_layers = 1
        self.zdim = args.zdim
        self.num_gaussians = 20
        self.dropout = 0.1

        # Maps the latent vector to an initial cell/hidden vector
        self.hidden_cell_2_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.zdim + self.stroke_emd_size, 2 * self.dec_rnn_size_2),
            torch.nn.Tanh()
        )

        self.lstm_2 = torch.nn.LSTM(
            self.stroke_emd_size + self.zdim + 5,
            self.dec_rnn_size_2,
            num_layers=self.num_layers,
            # dropout=self.dropout,
            batch_first=True)

        self.parameters_predictor = torch.nn.Linear(self.dec_rnn_size_2, 6 * self.num_gaussians + 3)

    def get_mixture_params(self, output):
        pen_logits = output[:, :, 0:3]
        pi, mu1, mu2, sigma1, sigma2, corr = torch.split(output[:, :, 3:], self.num_gaussians, dim=2)

        pi = torch.nn.functional.softmax(pi, dim=-1)
        pen = torch.nn.functional.softmax(pen_logits, dim=-1)

        sigma1 = sigma1.exp()
        sigma2 = sigma2.exp()
        corr = torch.nn.Tanh()(corr)

        return pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits

    def forward(self, seqs, z, pred_stroke_emb, hidden_and_cell_2=None):
        # bs, steps = inputs.shape[:2]
        bs, max_stroke_num = pred_stroke_emb.shape[0:2]
        max_stroke_len = seqs.shape[2]

        # lstm-1 to predict stroke and position embeddings
        expanded_z = z.unsqueeze(1).repeat(1, max_stroke_num, 1)
        seqs = seqs.contiguous().reshape(bs * max_stroke_num, max_stroke_len, 5)

        temp = torch.cat([pred_stroke_emb, expanded_z], dim=2).contiguous().reshape(bs * max_stroke_num, self.zdim + self.stroke_emd_size)
        inputs_2 = torch.cat([seqs, temp.unsqueeze(dim=1).repeat(1, max_stroke_len, 1)], dim=2)

        if hidden_and_cell_2 is None:
            hidden_and_cell_2 = self.hidden_cell_2_predictor(temp)
            hidden_2 = hidden_and_cell_2[:, :self.dec_rnn_size_2]
            hidden_2 = hidden_2.unsqueeze(0).contiguous()
            cell_2 = hidden_and_cell_2[:, self.dec_rnn_size_2:]
            cell_2 = cell_2.unsqueeze(0).contiguous()
            hidden_and_cell_2 = (hidden_2, cell_2)

        outputs_2, hidden_and_cell_2 = self.lstm_2(inputs_2, hidden_and_cell_2)

        # if self.training:
        # At train time we want parameters for each time step
        # outputs_2 = outputs_2.contiguous().reshape(bs * max_stroke_num * max_stroke_len, self.dec_rnn_size_2)
        params = self.parameters_predictor(outputs_2)  # [bs * max_stroke_num, max_stroke_len, d]
        pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits = self.get_mixture_params(params)

        return pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits, hidden_and_cell_2

class Reconstructor(torch.nn.Module):
    def __init__(self, args):
        super(Reconstructor, self).__init__()
        self.zdim = args.zdim

        self.fc = torch.nn.Linear(self.zdim, self.zdim * 4 * 4)

        self.reconstruct = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.contiguous().reshape(z.shape[0], self.zdim, 4, 4)
        cn1_out = self.reconstruct(torch.nn.ReLU()(out))
        return cn1_out

# class Reconstructor(torch.nn.Module):
#     '''lstm-2 to predict drawing actions'''
#     def __init__(self, args):
#         super(Reconstructor, self).__init__()
#         self.stroke_emd_size = args.stroke_emd_size
#         self.dec_rnn_size_2 = args.dec_rnn_size_2
#         self.num_layers = 1
#         self.zdim = args.zdim
#         self.num_gaussians = 20
#         self.dropout = 0.1
#
#         # Maps the latent vector to an initial cell/hidden vector
#         self.hidden_cell_2_predictor = torch.nn.Sequential(
#             torch.nn.Linear(self.zdim, 2 * self.dec_rnn_size_2),
#             torch.nn.Tanh()
#         )
#
#         self.lstm_2 = torch.nn.LSTM(
#             self.zdim + 5,
#             self.dec_rnn_size_2,
#             num_layers=self.num_layers,
#             # dropout=self.dropout,
#             batch_first=True)
#
#         self.parameters_predictor = torch.nn.Linear(self.dec_rnn_size_2, 6 * self.num_gaussians + 3)
#
#     def get_mixture_params(self, output):
#         pen_logits = output[:, 0:3]
#         pi, mu1, mu2, sigma1, sigma2, corr = torch.split(output[:, 3:], self.num_gaussians, dim=1)
#
#         pi = torch.nn.functional.softmax(pi, dim=-1)
#         pen = torch.nn.functional.softmax(pen_logits, dim=-1)
#
#         sigma1 = sigma1.exp()
#         sigma2 = sigma2.exp()
#         corr = torch.nn.Tanh()(corr)
#
#         return pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits
#
#     def forward(self, seqs, z, hidden_and_cell_2=None):
#         bs, max_seq_num = seqs.shape[0:2]
#
#         expanded_z = z.unsqueeze(1).repeat(1, max_seq_num, 1)
#         inputs_2 = torch.cat([seqs, expanded_z], dim=2)
#
#         if hidden_and_cell_2 is None:
#             hidden_and_cell_2 = self.hidden_cell_2_predictor(z)
#             hidden_2 = hidden_and_cell_2[:, :self.dec_rnn_size_2]
#             hidden_2 = hidden_2.unsqueeze(0).contiguous()
#             cell_2 = hidden_and_cell_2[:, self.dec_rnn_size_2:]
#             cell_2 = cell_2.unsqueeze(0).contiguous()
#             hidden_and_cell_2 = (hidden_2, cell_2)
#
#         outputs_2, hidden_and_cell_2 = self.lstm_2(inputs_2, hidden_and_cell_2)
#
#         # if self.training:
#         # At train time we want parameters for each time step
#         outputs_2 = outputs_2.contiguous().reshape(bs * max_seq_num, self.dec_rnn_size_2)
#         params = self.parameters_predictor(outputs_2)
#         pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits = self.get_mixture_params(params)
#
#         return pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits, hidden_and_cell_2
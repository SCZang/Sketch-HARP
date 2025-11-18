import argparse
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision

import six
import model
import random
from scipy.linalg import block_diag
from seq2img import make_graph_

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, categories, max_stroke_num, max_stroke_len, mode='train', normal_sketch_path=None):
        super(Dataset, self).__init__()

        self.mode = mode
        self.limit = 1000

        if self.mode not in ["train", "test", "valid"]:
            raise ValueError("Only allowed data modes are: 'train', 'test', 'valid',.")

        self.max_stroke_num = max_stroke_num
        self.max_stroke_len = max_stroke_len

        count = 0
        for ctg in categories:
            seq_path = os.path.join(data_dir, ctg + '.npz')
            if six.PY3:
                seq_data = np.load(seq_path, encoding='latin1', allow_pickle=True)
            else:
                seq_data = np.load(seq_path, allow_pickle=True)

            if count == 0:
                train_seqs = seq_data['train']
                valid_seqs = seq_data['valid']
                test_seqs = seq_data['test']
            else:
                train_seqs = np.concatenate((train_seqs, seq_data['train']))
                valid_seqs = np.concatenate((valid_seqs, seq_data['valid']))
                test_seqs = np.concatenate((test_seqs, seq_data['test']))
            count += 1

        self.max_seq_len = self.get_max_len(np.concatenate((train_seqs, valid_seqs, test_seqs)))

        if self.mode == 'train':
            self.strokes, self.seq_lens = self.preprocess_data(train_seqs)
        elif self.mode == 'valid':
            self.strokes, self.seq_lens = self.preprocess_data(valid_seqs)
        else:
            self.strokes, self.seq_lens = self.preprocess_data(test_seqs)

    # def preprocess_data(self, seqs):
    #     # pre-process
    #     strokes = []
    #     seq_lens = []
    #     scale_factor = self.calculate_normalizing_scale_factor(seqs)
    #
    #     count_data = 0  # the number of drawing with length less than N_max
    #     for i in range(len(seqs)):
    #         seq = np.copy(seqs[i])
    #         seq_len = len(seq)
    #         seq_lens.append(seq_len)
    #         if len(seq) <= self.max_seq_len:  # keep data with length less than N_max
    #             count_data += 1
    #             # removes large gaps from the data
    #             seq = np.minimum(seq, self.limit)  # prevent large values
    #             seq = np.maximum(seq, -self.limit)  # prevent small values
    #             seq = np.array(seq, dtype=float)  # change data type
    #             seq[:, 0:2] /= scale_factor  # scale the first two dims of data
    #             strokes.append(seq)
    #     return strokes, seq_lens

    def preprocess_data(self, seqs):
        # pre-process
        strokes = []
        seq_lens = []
        scale_factor = self.calculate_normalizing_scale_factor(seqs)

        count_data = 0  # the number of drawing with length less than N_max
        for i in range(len(seqs)):
            tmp_abs = np.copy(seqs[i])
            tmp_abs[:, :2] = np.cumsum(seqs[i][:, :2], axis=0)
            abs_seq = self.sketch_normalize(tmp_abs)
            # abs_seq = np.concatenate([np.zeros([1, 3]), abs_seq], axis=0)

            # abs to rel
            seq_1 = abs_seq[:, :2].copy()
            seq_0 = np.concatenate([np.zeros([1, 2]), seq_1[:-1].copy()], axis=0)
            seq = np.concatenate([seq_1 - seq_0, abs_seq[:, 2:].copy()], axis=1)
            # seq[0, 0:2] = 0

            seq_len = len(seq)
            seq_lens.append(seq_len)
            if len(seq) <= self.max_seq_len:    # keep data with length less than N_max
                count_data += 1
                # removes large gaps from the data
                # seq = np.minimum(seq, self.limit)     # prevent large values
                # seq = np.maximum(seq, -self.limit)    # prevent small values
                # seq = np.array(seq, dtype=float)  # change data type
                # seq[:, 0:2] /= scale_factor       # scale the first two dims of data
                strokes.append(seq)
        return strokes, seq_lens

    def sketch_normalize(self, sketch):
        '''

        :param sketch: (points, abs x, abs y, state)
        :return: normalize(sketch)
        '''
        n_sketch = sketch.copy().astype(np.float32)
        w_max = np.max(n_sketch[:, 0], axis=0)
        w_min = np.min(n_sketch[:, 0], axis=0)

        h_max = np.max(n_sketch[:, 1], axis=0)
        h_min = np.min(n_sketch[:, 1], axis=0)

        scale_w = w_max - w_min
        scale_h = h_max - h_min

        scale = scale_w if scale_w > scale_h else scale_h

        n_sketch[:, 0] = (n_sketch[:, 0] - w_min) / scale * 2 - 1
        n_sketch[:, 1] = (n_sketch[:, 1] - h_min) / scale * 2 - 1
        return n_sketch

    def get_max_len(self, strokes):
        max_len = 0
        for stroke in strokes:
            ml = len(stroke)
            if ml > max_len:
                max_len = ml
        return max_len

    def get_max_stroke_num(self, all_seqs):
        max_count = 0
        for seq in all_seqs:
            strokes = self.split_strokes(seq)
            max_count = max(max_count, len(strokes))
        return max_count

    def get_max_stroke_len(self, all_seqs):
        max_len = 0
        for seq in all_seqs:
            strokes = self.split_strokes(seq)
            for stroke in strokes:
                if len(stroke) > max_len:
                    max_len = len(stroke)
                # max_len = max(max_len, len(stroke))

        return max_len + 1

    def calculate_normalizing_scale_factor(self, sketches):
        data = []
        for sketch in sketches:
            for stroke in sketch:
                data.append(stroke)
        return np.std(np.array(data))

    def random_scale_seq(self, data):
        """ Augment data by stretching x and y axis randomly [1-e, 1+e] """
        random_scale_factor = 0.1
        x_scale_factor = (np.random.random() - 0.5) * 2 * random_scale_factor + 1.0
        y_scale_factor = (np.random.random() - 0.5) * 2 * random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result

    def seq_3d_to_5d(self, stroke, max_len=250):
        """ Convert from 3D format (npz file) to 5D (sketch-rnn paper) """
        result = np.zeros((max_len + 1, 5), dtype=float)
        l = len(stroke)
        assert l <= max_len
        result[0:l, 0:2] = stroke[:, 0:2]
        result[0:l, 3] = stroke[:, 2]
        result[0:l, 2] = 1 - result[0:l, 3]
        result[l:, 4] = 1

        # put in the first token, as described in sketch-rnn methodology
        start_stroke_token = [0, 0, 1, 0, 0]  # S_0 in sketch-rnn paper
        result[1:, :] = result[:-1, :]
        result[0, :] = 0
        result[0, 2] = start_stroke_token[2]  # setting S_0 from paper.
        result[0, 3] = start_stroke_token[3]
        result[0, 4] = start_stroke_token[4]
        return result

    # def seq_3d_to_5d(self, stroke, max_len=250):
    #     """ Convert from 3D format (npz file) to 5D (sketch-rnn paper) """
    #     result = np.zeros((max_len, 5), dtype=float)
    #     l = len(stroke)
    #     assert l <= max_len
    #     result[0:l, 0:2] = stroke[:, 0:2]
    #     result[0:l, 3] = stroke[:, 2]
    #     result[0:l, 2] = 1 - result[0:l, 3]
    #     result[l:, 4] = 1
    #     return result

    def shuffle_single_strokes(self, data):
        # split_indices = np.where(data[:, 2] == 1)[0] + 1
        # strokes = np.split(data, split_indices)
        #
        # strokes = [s for s in strokes if len(s) > 0]

        strokes = self.split_strokes(data)

        abs_strokes = []
        abs_x, abs_y = 0, 0
        for stroke in strokes:
            new_stroke = np.copy(stroke)
            for i in range(len(new_stroke)):
                abs_x += new_stroke[i, 0]
                abs_y += new_stroke[i, 1]
                new_stroke[i, 0] = abs_x
                new_stroke[i, 1] = abs_y
            abs_strokes.append(new_stroke)

        random.shuffle(abs_strokes)

        shuffled_data = []
        prev_x, prev_y = 0, 0
        for stroke in abs_strokes:
            new_stroke = np.copy(stroke)
            for i in range(len(new_stroke)):
                temp_x, temp_y = new_stroke[i, 0], new_stroke[i, 1]
                new_stroke[i, 0] -= prev_x
                new_stroke[i, 1] -= prev_y
                prev_x, prev_y = temp_x, temp_y
            shuffled_data.append(new_stroke)

        return np.concatenate(shuffled_data, axis=0)

    def split_strokes(self, seq):
        # strokes = []
        # start_idx = 0
        # for i in range(len(seq)):
        #     if seq[i, 2] == 1:
        #         strokes.append(seq[start_idx:i + 1])
        #         start_idx = i + 1
        split_indices = np.where(seq[:, 2] == 1)[0] + 1
        strokes = np.split(seq, split_indices)
        strokes = [s for s in strokes if len(s) > 0]

        return strokes

    def strokes_3d_to_5d_and_pad(self, seq, max_stroke_num=50, max_stroke_len=150):
        ''' new ver.: with fixed max_stroke_num and max_stroke_len'''
        split = self.split_strokes(seq)
        padded_strokes = np.zeros((max_stroke_num, max_stroke_len, 5), dtype=float)
        padded_strokes[:, :, 4] = 1
        abs_start_positions = []
        stroke_lens = np.zeros((max_stroke_num,), dtype=int)

        full_abs_positions = []
        abs_pos = np.array([0.0, 0.0])
        for stroke in split:
            full_abs_positions.append(abs_pos.copy())
            abs_pos += np.sum(stroke[:, :2], axis=0)

        stroke_num = min(len(split), max_stroke_num)

        for i in range(stroke_num):
            stroke = split[i]
            start_pos = full_abs_positions[i]
            abs_start_positions.append(start_pos)

            l = len(stroke)
            if l + 1 > max_stroke_len:
                stroke = stroke[:max_stroke_len - 1]
                l = len(stroke)

            result = np.zeros((max_stroke_len, 5), dtype=float)
            result[1:l+1, 0:2] = stroke[:, 0:2]
            result[1:l+1, 3] = stroke[:, 2]
            result[1:l+1, 2] = 1 - result[1:l+1, 3]
            result[l+1:, 4] = 1
            result[0, :] = [0, 0, 1, 0, 0]

            padded_strokes[i] = result
            stroke_lens[i] = l

        while len(abs_start_positions) < max_stroke_num:
            abs_start_positions.append(np.array([0.0, 0.0]))

        abs_start_positions = np.stack(abs_start_positions)
        return padded_strokes, abs_start_positions, stroke_lens, stroke_num

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        seq_len = self.seq_lens[idx]
        # if self.mode == 'train':
        #     strokes_3d = self.shuffle_single_strokes(self.strokes[idx])
        # else:
        #     strokes_3d = self.strokes[idx]
        strokes_3d = self.strokes[idx]
        strokes_5d = self.seq_3d_to_5d(strokes_3d, self.max_seq_len)

        # padded_strokes_5d: [max_stroke_num, max_stroke_len, 5]
        padded_strokes_5d, abs_start_positions, stroke_lens, stroke_nums = self.strokes_3d_to_5d_and_pad(strokes_3d,
                                                                                                         self.max_stroke_num,
                                                                                                         self.max_stroke_len)

        seed = np.load('/workspace/dataset/random_seed.npy', allow_pickle=True)
        seed_id = 0
        graph_number = 3
        #############################
        strokes_3d[0, 0:2] = 0
        #############################
        store_pen_location, graph, adj, _graph_len, mask_id, seed_id = make_graph_(strokes_3d, seed, seed_id,
                                                                                   graph_num=graph_number,
                                                                                   graph_picture_size=128,
                                                                                   mask_prob=0.0, train=self.mode)

        return strokes_5d.astype(np.float32), seq_len, padded_strokes_5d.astype(np.float32), abs_start_positions.astype(np.float32), stroke_lens, stroke_nums, graph.astype(np.float32)
        # # transform sketch sequences to images
        # seed = np.load('/data/datasets/quickdraw/random_seed.npy', allow_pickle=True)
        # seed_id = 0
        # graph_number = 20
        # store_pen_location, graph, adj, _graph_len, mask_id, seed_id = make_graph_(strokes_3d_1, seed, seed_id,
        #                                                                            graph_num=graph_number,
        #                                                                            graph_picture_size=128,
        #                                                                            mask_prob=0.0, train=self.mode)
        # if _graph_len == (graph_number - 1):
        #     adj_mask = np.ones([graph_number - 1, graph_number - 1])
        # else:
        #     adj_mask = np.concatenate([np.concatenate([np.ones([_graph_len + 1, _graph_len + 1]),
        #                                                np.zeros([graph_number - 2 - _graph_len, _graph_len + 1])], axis=0),
        #                                np.zeros([graph_number - 1, graph_number - 2 - _graph_len])], axis=1)
        # for id in mask_id:
        #     adj_mask[id, :] = 0
        #     adj_mask[:, id] = 0
        # return strokes_5d_1.astype(np.float32), strokes_5d_2.astype(np.float32), seq_len, graph.astype(np.float32), adj_mask.astype(np.float32)

class GaussianMixtureReconstructionLoss(torch.nn.Module):
    def __init__(self, bs, eps=1e-6):
        super(GaussianMixtureReconstructionLoss, self).__init__()
        self.bs = bs
        self.eps = eps

    def get_density(self, x1, x2, mu1, mu2, s1, s2, rho):
        norm1 = x1 - mu1
        norm2 = x2 - mu2
        s1s2 = s1 * s2
        z = (norm1 / (s1 + self.eps)).square() + (norm2 / (s2 + self.eps)).square() - 2. * rho * norm1 * norm2 / (s1s2 + self.eps)
        neg_rho = 1. - rho.square()
        result = (-z / (2. * neg_rho + self.eps)).exp()
        denom = 2 * np.pi * s1s2 * neg_rho.sqrt()
        result = result / (denom + self.eps)
        return result

    def forward(self, pi, mu1, mu2, s1, s2, corr, pen_logits, x1_data, x2_data, pen_data, marker, mode):
        stroke_masks = torch.eq(marker[:, :, 0], 1).cuda().float()
        stroke_masks = stroke_masks.reshape(-1)  # [bs * max_stroke_num]f

        result0 = self.get_density(x1_data, x2_data, mu1, mu2, s1, s2, corr)
        result1 = (result0 * pi).sum(dim=2)  # [bs * max_stroke_num, max_stroke_len]
        result1 = -(result1 + self.eps).log()  # Avoid log(0)

        masks = 1.0 - pen_data[:, :, 2]  # [bs * max_stroke_num, max_stroke_len]
        result1 = (result1 * masks).sum(dim=1)  # [bs * max_stroke_num]
        result2 = torch.nn.functional.cross_entropy(pen_logits.permute(0, 2, 1), pen_data.permute(0, 2, 1), reduction='none')  # [bs * max_stroke_num, max_stroke_len]
        if mode == 'train':
            result2 = result2.sum(dim=1)
        else:
            result2 = (result2 * masks).sum(dim=1)
        return ((result1 + result2) * stroke_masks).sum() / self.bs


class StrokeNumPredictLoss(torch.nn.Module):
    def __init__(self):
        super(StrokeNumPredictLoss, self).__init__()

    def forward(self, logits, marker, mode):
        masks = torch.eq(marker[:, :, 0], 1).float()
        result = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), marker.reshape(-1, 2).to(logits.device), reduction='none')
        if mode == 'train':
            result = result.sum()
        else:
            result = (result * masks.reshape(-1).to(logits.device)).sum()
        return result / len(marker)

class StrokeEmbReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super(StrokeEmbReconstructionLoss, self).__init__()

    def forward(self, pred_stroke_emb, gt_stroke_emb, marker):
        masks = torch.eq(marker[:, :, 0], 1).float()
        result1 = ((pred_stroke_emb - gt_stroke_emb).pow(2).sum(dim=2) * masks.to(pred_stroke_emb.device)).sum()
        return result1 / len(pred_stroke_emb)

class PosEmbReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super(PosEmbReconstructionLoss, self).__init__()
        self.eps = 1e-6

    def get_density(self, x1, x2, mu1, mu2, s1, s2, rho):
        norm1 = x1 - mu1
        norm2 = x2 - mu2
        s1s2 = s1 * s2
        z = (norm1 / (s1 + self.eps)).square() + (norm2 / (s2 + self.eps)).square() - 2. * rho * norm1 * norm2 / (s1s2 + self.eps)
        neg_rho = 1. - rho.square()
        result = (-z / (2. * neg_rho + self.eps)).exp()
        denom = 2 * np.pi * s1s2 * neg_rho.sqrt()
        result = result / (denom + self.eps)
        return result

    def forward(self, mu1, mu2, s1, s2, corr, gt_pos, marker):
        x1_data, x2_data = torch.split(gt_pos, 1, dim=2)
        result0 = self.get_density(x1_data, x2_data, mu1, mu2, s1, s2, corr)
        result1 = result0.squeeze(dim=2)
        result1 = -(result1 + self.eps).log()  # Avoid log(0)

        masks = torch.eq(marker[:, :, 0], 1).float()
        result1 = (result1 * masks.to(mu1.device)).sum()
        return result1 / len(mu1)

class Model:
    def __init__(self):
        self.seq_encoder: nn.Module = model.SeqEncoder(args=args).cuda()
        self.stroke_encoder: nn.Module = model.StrokeEncoder(args=args).cuda()
        self.stroke_decoder: nn.Module = model.StrokeDecoder(args=args).cuda()
        self.seq_decoder: nn.Module = model.SeqDecoder(args=args).cuda()
        self.pos_decoder: nn.Module = model.PosDecoder(args=args).cuda()
        self.rec_decoder: nn.Module = model.Reconstructor(args=args).cuda()

        self.lil_loss = GaussianMixtureReconstructionLoss(args.bs)
        self.stroke_emd_loss = StrokeEmbReconstructionLoss()
        self.pos_emd_loss = PosEmbReconstructionLoss()
        self.stopper_loss = StrokeNumPredictLoss()

        self.enc_optimizer = torch.optim.Adam([{'params': self.seq_encoder.parameters()},
                                               {'params': self.stroke_encoder.parameters()}], args.lr)
        self.dec_optimizer = torch.optim.Adam([{'params': self.stroke_decoder.parameters()},
                                               {'params': self.seq_decoder.parameters()},
                                               {'params': self.pos_decoder.parameters()},
                                               {'params': self.rec_decoder.parameters()}], args.lr)

    def load(self, seq_enc_name, stroke_enc_name, stroke_dec_name, seq_dec_name, pos_dec_name, rec_dec_name):
        saved_seq_enc = torch.load(seq_enc_name)
        saved_stroke_enc = torch.load(stroke_enc_name)
        saved_stroke_dec = torch.load(stroke_dec_name)
        saved_seq_dec = torch.load(seq_dec_name)
        saved_rec_dec = torch.load(rec_dec_name)
        saved_pos_dec = torch.load(pos_dec_name)
        self.seq_encoder.load_state_dict(saved_seq_enc)
        self.stroke_encoder.load_state_dict(saved_stroke_enc)
        self.stroke_decoder.load_state_dict(saved_stroke_dec)
        self.seq_decoder.load_state_dict(saved_seq_dec)
        self.pos_decoder.load_state_dict(saved_pos_dec)
        self.rec_decoder.load_state_dict(saved_rec_dec)

    def save(self, epoch):
        torch.save(self.seq_encoder.state_dict(), f'./sketch_model/seq_enc_epoch_{epoch}.pth')
        torch.save(self.stroke_encoder.state_dict(), f'./sketch_model/stroke_enc_epoch_{epoch}.pth')
        torch.save(self.stroke_decoder.state_dict(), f'./sketch_model/stroke_dec_epoch_{epoch}.pth')
        torch.save(self.seq_decoder.state_dict(), f'./sketch_model/seq_dec_epoch_{epoch}.pth')
        torch.save(self.pos_decoder.state_dict(), f'./sketch_model/pos_dec_epoch_{epoch}.pth')
        torch.save(self.rec_decoder.state_dict(), f'./sketch_model/rec_dec_epoch_{epoch}.pth')

    def lr_decay(self, optimizer):
        """Decay learning rate by a factor of lr_decay"""
        for param_group in optimizer.param_groups:
            if param_group['lr'] > 1e-6:
                param_group['lr'] *= 0.9
        return optimizer

    def train(self, epoch, dataloader):
        self.seq_encoder.train()
        self.stroke_encoder.train()
        self.stroke_decoder.train()
        self.seq_decoder.train()
        self.pos_decoder.train()
        self.rec_decoder.train()

        step = 0
        start = time.time()
        for batch in dataloader:
            if step == 2500:
                break

            seqs, seq_lens, padded_strokes, start_positions, stroke_lens, stroke_nums, graphs = batch
            # seqs = seqs.cuda().float()
            # seq_lens = seq_lens.long()
            padded_strokes = padded_strokes.cuda().float()
            start_positions = start_positions.cuda().float()
            stroke_lens = stroke_lens.long()
            stroke_nums = stroke_nums.long()
            graphs = graphs.cuda().float()

            # import cv2
            # cv2.imwrite("12.png", (graphs[0, 0].cpu().numpy() + 1)/2*255)
            # exit(0)


            bs, num, len = padded_strokes.shape[0:3]
            # an ont-hot marker to reveal when no stroke is needed to finish a sketch
            # 1 and 0 indicate the position with a stroke and empty padding, respectively
            stroke_stop_marker = torch.zeros([bs, num, 2])
            for i in range(bs):
                stroke_stop_marker[i, 0:stroke_nums[i], 0] = 1
                stroke_stop_marker[i, stroke_nums[i]:, 1] = 1

            rel_start_positions = torch.zeros_like(start_positions).cuda().float()
            rel_start_positions[:, 1:] = start_positions[:, 1:] - start_positions[:, :-1]
            rel_start_positions = rel_start_positions * stroke_stop_marker[:, :, 0].unsqueeze(dim=2).cuda().float()

            stroke_emb, pos_emb = self.seq_encoder(padded_strokes, start_positions, stroke_lens + 1, stroke_stop_marker[:, :, 0])
            seq_z = self.stroke_encoder(stroke_emb, pos_emb, stroke_nums)

            _, _, emb_size = stroke_emb.shape
            padded_stroke_emb = torch.cat([-1 * torch.ones([bs, 1, emb_size]).to(seq_z.device), stroke_emb[:, :-1]], dim=1)  # assign the starting marks
            padded_pos_emb = torch.cat([-1 * torch.ones([bs, 1, emb_size]).to(seq_z.device), pos_emb[:, :-1]], dim=1)
            pred_stroke_emb, stopper_logits, pred_stopper, _ = self.stroke_decoder(seq_z, padded_stroke_emb, padded_pos_emb)

            pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits, _ = self.seq_decoder(padded_strokes[:, :, :-1], seq_z, stroke_emb)
            pos_mu1, pos_mu2, pos_sigma1, pos_sigma2, pos_corr, _ = self.pos_decoder(seq_z, stroke_emb, pos_emb[:, :-1])

            rec_seq = self.rec_decoder(seq_z)
            # rec_pi, rec_mu1, rec_mu2, rec_sigma1, rec_sigma2, rec_corr, rec_pen, rec_pen_logits, _ = self.rec_decoder(seqs[:, :-1], seq_z)

            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

            # x1_data, x2_data, pen_data = torch.split(padded_strokes[:, :, 1:].contiguous().reshape(-1, 5), [1, 1, 3], dim=-1)
            x1_data, x2_data, pen_data = torch.split(padded_strokes[:, :, 1:].contiguous().reshape(bs * num, len - 1, 5), [1, 1, 3], dim=-1)

            lil_loss = self.lil_loss(pi, mu1, mu2, sigma1, sigma2, corr, pen_logits, x1_data, x2_data, pen_data, stroke_stop_marker, "train")
            stroke_emd_loss = self.stroke_emd_loss(pred_stroke_emb, stroke_emb.detach(), stroke_stop_marker)
            pos_emb_loss = self.pos_emd_loss(pos_mu1, pos_mu2, pos_sigma1, pos_sigma2, pos_corr, rel_start_positions[:, 1:], stroke_stop_marker[:, 1:])
            # pos_emb_loss = ((pred_pos - start_positions[:, 1:]).pow(2).sum(dim=-1) * torch.eq(stroke_stop_marker[:, 1:, 0], 1).float().to(seq_z.device)).sum() / seq_z.shape[0]
            stopper_loss = self.stopper_loss(stopper_logits, stroke_stop_marker, "train")
            image = graphs[:, 0].permute(0, 3, 1, 2)
            rec_loss = (rec_seq - image).pow(2).mean()
            # rec_x1_data, rec_x2_data, rec_pen_data = torch.split(seqs[:, 1:].contiguous().reshape(-1, 5), [1, 1, 3], dim=-1)
            # rec_loss = self.lil_loss(rec_pi, rec_mu1, rec_mu2, rec_sigma1, rec_sigma2, rec_corr, rec_pen_logits, rec_x1_data, rec_x2_data, rec_pen_data, "train")

            # import cv2
            # print(image.max(), image.min(), rec_seq.max(), rec_seq.min())
            # cv2.imwrite('gt.png', (1 + image[0].permute(1, 2, 0).cpu().numpy())/2*255)
            # cv2.imwrite('fake.png', (1+rec_seq[0].permute(1, 2, 0).cpu().detach().numpy())/2*255)
            # exit(0)

            loss = lil_loss + 10 * stroke_emd_loss + pos_emb_loss + stopper_loss + 0.5 * rec_loss
            loss.backward()

            grad_clip = 1.
            torch.nn.utils.clip_grad_norm_(self.seq_encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.stroke_encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.stroke_decoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.seq_decoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.pos_decoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.rec_decoder.parameters(), grad_clip)

            self.enc_optimizer.step()
            self.dec_optimizer.step()

            if (step % 20) == 0:
                end = time.time()
                time_taken = end - start
                start = time.time()

                print("Epoch: %d, Step: %d, LR: %.5f, Lil: %.2f, Stroke: %.2f, Pos: %.2f, Stop: %.2f, Rec: %.2f, Time: %.1f"
                      % (epoch, step, self.enc_optimizer.param_groups[0]["lr"], lil_loss, stroke_emd_loss, pos_emb_loss, stopper_loss, rec_loss, time_taken))
            step += 1

        self.enc_optimizer = self.lr_decay(self.enc_optimizer)
        self.dec_optimizer = self.lr_decay(self.dec_optimizer)

    def eval(self, dataloader):
        self.seq_encoder.eval()
        self.stroke_encoder.eval()
        self.stroke_decoder.eval()
        self.seq_decoder.eval()
        self.pos_decoder.eval()
        self.rec_decoder.eval()

        loss_sum = 0
        lil_loss_sum = 0
        stroke_emd_loss_sum = 0
        pos_emb_loss_sum = 0
        stopper_loss_sum = 0
        count = 0
        for batch in dataloader:
            seqs, seq_lens, padded_strokes, start_positions, stroke_lens, stroke_nums, graphs = batch
            seqs = seqs.cuda().float()
            # seq_lens = seq_lens.long()
            padded_strokes = padded_strokes.cuda().float()
            start_positions = start_positions.cuda().float()
            stroke_lens = stroke_lens.long()
            stroke_nums = stroke_nums.long()
            # graphs = graphs.cuda().float()

            with torch.no_grad():
                bs, num, len = padded_strokes.shape[0:3]
                # an ont-hot marker to reveal when no stroke is needed to finish a sketch
                # 1 and 0 indicate the position with a stroke and empty padding, respectively
                stroke_stop_marker = torch.zeros([bs, num, 2])
                for i in range(bs):
                    stroke_stop_marker[i, 0:stroke_nums[i], 0] = 1
                    stroke_stop_marker[i, stroke_nums[i]:, 1] = 1

                rel_start_positions = torch.zeros_like(start_positions).cuda().float()
                rel_start_positions[:, 1:] = start_positions[:, 1:] - start_positions[:, :-1]
                rel_start_positions = rel_start_positions * stroke_stop_marker[:, :, 0].unsqueeze(dim=2).cuda().float()

                stroke_emb, pos_emb = self.seq_encoder(padded_strokes, start_positions, stroke_lens + 1, stroke_stop_marker[:, :, 0])
                seq_z = self.stroke_encoder(stroke_emb, pos_emb, stroke_nums)

                _, _, emb_size = stroke_emb.shape
                padded_stroke_emb = torch.cat([-1 * torch.ones([bs, 1, emb_size]).to(seq_z.device), stroke_emb[:, :-1]], dim=1)  # assign the starting marks
                padded_pos_emb = torch.cat([-1 * torch.ones([bs, 1, emb_size]).to(seq_z.device), pos_emb[:, :-1]], dim=1)
                pred_stroke_emb, stopper_logits, pred_stopper, _ = self.stroke_decoder(seq_z, padded_stroke_emb, padded_pos_emb)
                pos_mu1, pos_mu2, pos_sigma1, pos_sigma2, pos_corr, _ = self.pos_decoder(seq_z, pred_stroke_emb, pos_emb[:, :-1])
                pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits, _ = self.seq_decoder(padded_strokes[:, :, :-1], seq_z, pred_stroke_emb)
                # rec_seq = self.rec_decoder(seq_z)
                # rec_pi, rec_mu1, rec_mu2, rec_sigma1, rec_sigma2, rec_corr, rec_pen, rec_pen_logits, _ = self.rec_decoder(seqs[:, :-1], seq_z)

                # x1_data, x2_data, pen_data = torch.split(padded_strokes[:, :, 1:].contiguous().reshape(-1, 5), [1, 1, 3], dim=-1)
                x1_data, x2_data, pen_data = torch.split(padded_strokes[:, :, 1:].contiguous().reshape(bs * num, len - 1, 5), [1, 1, 3], dim=-1)

                lil_loss = self.lil_loss(pi, mu1, mu2, sigma1, sigma2, corr, pen_logits, x1_data, x2_data, pen_data, stroke_stop_marker, "eval")
                stroke_emd_loss = self.stroke_emd_loss(pred_stroke_emb, stroke_emb.detach(), stroke_stop_marker)
                pos_emb_loss = self.pos_emd_loss(pos_mu1, pos_mu2, pos_sigma1, pos_sigma2, pos_corr, rel_start_positions[:, 1:], stroke_stop_marker[:, 1:])
                # pos_emb_loss = ((pred_pos - start_positions[:, 1:]).pow(2).sum(dim=-1) * torch.eq(stroke_stop_marker[:, 1:, 0], 1).float().to(seq_z.device)).sum() / seq_z.shape[0]
                stopper_loss = self.stopper_loss(stopper_logits, stroke_stop_marker, "eval")

                loss_sum += lil_loss + stroke_emd_loss + pos_emb_loss + stopper_loss
                # loss_sum += lil_loss
                lil_loss_sum += lil_loss
                stroke_emd_loss_sum += stroke_emd_loss
                pos_emb_loss_sum += pos_emb_loss
                stopper_loss_sum += stopper_loss
                count += 1
        # print(lil_loss_sum/count, stroke_emd_loss_sum/count, pos_emb_loss_sum/count, stopper_loss_sum/count)
        print("Test: Lil: %.2f, Stroke: %.2f, Pos: %.2f, Stop: %.2f" % (lil_loss_sum/count, stroke_emd_loss_sum/count, pos_emb_loss_sum/count, stopper_loss_sum/count))
        return loss_sum / count

def train_model(args):
    trainset = Dataset(args.data_dir, args.categories, args.max_stroke_num, args.max_stroke_len, mode="train")
    validset = Dataset(args.data_dir, args.categories, args.max_stroke_num, args.max_stroke_len, mode="valid")
    train_dataloader = DataLoader(trainset, batch_size=args.bs, num_workers=16, shuffle=True)
    valid_dataloader = DataLoader(validset, batch_size=args.bs, num_workers=4, shuffle=False)


    model = Model()

    best_lil = 1e6

    if args.epoch_load != 0:
        model.load(f'./sketch_model/seq_enc_epoch_{args.epoch_load}.pth',
                   f'./sketch_model/stroke_enc_epoch_{args.epoch_load}.pth',
                   f'./sketch_model/stroke_dec_epoch_{args.epoch_load}.pth',
                   f'./sketch_model/seq_dec_epoch_{args.epoch_load}.pth',
                   f'./sketch_model/pos_dec_epoch_{args.epoch_load}.pth',
                   f'./sketch_model/rec_dec_epoch_{args.epoch_load}.pth')
        best_lil = np.load(f'./sketch_model/best_lil.npy')

    for epoch in range(args.num_epochs):
        if epoch < args.epoch_load:  # initialize the starting lr
            model.enc_optimizer = model.lr_decay(model.enc_optimizer)
            model.dec_optimizer = model.lr_decay(model.dec_optimizer)
            continue
        model.train(epoch, train_dataloader)
        current_lil = model.eval(valid_dataloader)
        print("Epoch: %d, Best-Lil: %.2f, Current-Lil: %.2f" % (epoch, best_lil, current_lil))

        if True:#current_lil.cpu().numpy() < best_lil:
            best_lil = current_lil.cpu().numpy()
            np.save(f'./sketch_model/best_lil.npy', current_lil.cpu().numpy())
            print("Model %d saved." % epoch)
            model.save(epoch)

class HParams:
    def __init__(self):
        self.data_dir = "/workspace/dataset"
        # self.categories = ["bee", "bus", "flower", "giraffe", "pig"]
        self.categories = ["airplane", "angel", "alarm_clock", "apple", "butterfly", "belt", "bus",
                           "cake", "cat", "clock", "eye", "fish", "pig", "sheep", "spider", "great_wall", "umbrella"]
        self.enc_rnn_size_1 = 512
        self.enc_rnn_size_2 = 512
        self.stroke_emd_size = 128   # size of stroke embedding
        self.dec_rnn_size_1 = 1024  # for stroke decoder
        self.dec_rnn_size_2 = 1024  # for sequence decoder
        self.dec_rnn_size_3 = 1024  # for position decoder
        self.max_stroke_num = 25
        self.max_stroke_len = 32
        self.zdim = 128
        self.lr = 0.001
        self.bs = 128
        self.num_epochs = 100
        self.epoch_load = 0

if __name__ == "__main__":
    args = HParams()
    train_model(args)

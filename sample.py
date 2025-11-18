import argparse
import os, glob, re
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
import model
from train import Dataset, HParams
from seq2img import make_graph_, seq_5d_to_3d
import cv2
from sklearn.manifold import TSNE
from tsne_clustering import arrangeTsneByDist, saveImage2Path

EPOCH_LOAD = 4
NUM_PER_CATEGORY = 2500

def sort_paths(paths):
    idxs = []
    for path in paths:
        idxs.append(int(re.findall(r'\d+', path)[-1]))

    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            if idxs[i] > idxs[j]:
                tmp = idxs[i]
                idxs[i] = idxs[j]
                idxs[j] = tmp

                tmp = paths[i]
                paths[i] = paths[j]
                paths[j] = tmp
    return paths

def split_strokes(seq):
    split_indices = np.where(seq[:, 2] == 1)[0] + 1
    strokes = np.split(seq, split_indices)
    strokes = [s for s in strokes if len(s) > 0]
    return strokes

def strokes_3d_to_5d_and_pad(seq, max_stroke_num=50, max_stroke_len=150):
    ''' new ver.: with fixed max_stroke_num and max_stroke_len'''
    split = split_strokes(seq)
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

def translate(out_seqs, args):
    strokes_3d = seq_5d_to_3d(out_seqs[0])
    padded_strokes_5d, abs_start_positions, stroke_lens, stroke_nums = strokes_3d_to_5d_and_pad(strokes_3d,
                                                                                                args.max_stroke_num,
                                                                                                args.max_stroke_len)
    return np.expand_dims(padded_strokes_5d, 0), np.expand_dims(abs_start_positions, 0), np.expand_dims(stroke_lens, 0), np.expand_dims(stroke_nums, 0)

def padded_5d_to_strokes_5d(out_padded_seqs, out_pos):
    if isinstance(out_padded_seqs, torch.Tensor):
        out_padded_seqs = out_padded_seqs[0].cpu().numpy()
    elif isinstance(out_padded_seqs, np.ndarray):
        out_padded_seqs = out_padded_seqs

    if isinstance(out_pos, torch.Tensor):
        out_pos = out_pos[0].cpu().numpy()
    elif isinstance(out_pos, np.ndarray):
        out_pos = out_pos

    # visualize_strokes(out_padded_seqs, out_pos)

    strokes = []
    for i in range(out_padded_seqs.shape[0]):
        stroke = out_padded_seqs[i]  # [65, 5]
        start_pos = out_pos[i]  # [2]

        valid_points = []
        for j in range(1, stroke.shape[0]):
            if stroke[j, 4] == 1:
                break
            valid_points.append(stroke[j])

        if len(valid_points) == 0:
            continue

        valid_points = np.stack(valid_points, axis=0)

        if valid_points[-1, 3] != 1:
            valid_points[-1, 3] = 1
            valid_points[-1, 2] = 0

        xy_rel = valid_points[:, 0:2]
        xy_abs = np.cumsum(xy_rel, axis=0) + start_pos
        valid_points[:, 0:2] = xy_abs

        strokes.append(valid_points)

    full = []
    prev_abs_end = None
    for i, stroke in enumerate(strokes):
        stroke_abs = stroke.copy()

        stroke_rel = np.zeros_like(stroke_abs)

        if prev_abs_end is None:
            stroke_rel[0, 0:2] = stroke_abs[0, 0:2]
        else:
            stroke_rel[0, 0:2] = stroke_abs[0, 0:2] - prev_abs_end

        stroke_rel[0, 2:] = stroke_abs[0, 2:]

        for j in range(1, stroke_abs.shape[0]):
            stroke_rel[j, 0:2] = stroke_abs[j, 0:2] - stroke_abs[j - 1, 0:2]
            stroke_rel[j, 2:] = stroke_abs[j, 2:]

        prev_abs_end = stroke_abs[-1, 0:2]
        full.append(stroke_rel)

    final_seq = np.concatenate(full, axis=0)
    return final_seq[None, :, :]

class Eval_Model:
    def __init__(self, args):
        self.seq_encoder: nn.Module = model.SeqEncoder(args=args).cuda()
        self.stroke_encoder: nn.Module = model.StrokeEncoder(args=args).cuda()
        self.stroke_decoder: nn.Module = model.StrokeDecoder(args=args).cuda()
        self.pos_decoder: nn.Module = model.PosDecoder(args=args).cuda()
        self.seq_decoder: nn.Module = model.SeqDecoder(args=args).cuda()

        self.args = args

    def load(self, seq_enc_name, stroke_enc_name, stroke_dec_name, seq_dec_name, pos_dec_name):
        saved_seq_enc = torch.load(seq_enc_name)
        saved_stroke_enc = torch.load(stroke_enc_name)
        saved_stroke_dec = torch.load(stroke_dec_name)
        saved_seq_dec = torch.load(seq_dec_name)
        saved_pos_dec = torch.load(pos_dec_name)
        self.seq_encoder.load_state_dict(saved_seq_enc)
        self.stroke_encoder.load_state_dict(saved_stroke_enc)
        self.stroke_decoder.load_state_dict(saved_stroke_dec)
        self.seq_decoder.load_state_dict(saved_seq_dec)
        self.pos_decoder.load_state_dict(saved_pos_dec)

    def adjust_pdf(self, pi_pdf, temp):
        pi_pdf = np.log(pi_pdf + 1e-10) / temp
        pi_pdf -= np.max(pi_pdf)
        pi_pdf = np.exp(pi_pdf)
        pi_pdf /= np.sum(pi_pdf)
        return pi_pdf

    def get_pi_idx(self, x, pdf, temp=1.0, greedy=False):
        """ Sample from a pdf, optionally greedily """
        if greedy:
            return np.argmax(pdf)
        pdf = self.adjust_pdf(np.copy(pdf), temp)
        accumulate = 0
        for i in range(0, pdf.size):
            accumulate += pdf[i]
            if accumulate >= x:
                return i
        return -1

    def sample_gaussian_2d(self, mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
        """ Sample from a 2D Gaussian """
        if greedy:
            return mu1, mu2
        mean = [mu1, mu2]
        s1 *= temp * temp
        s2 *= temp * temp
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    def sample(self, z, args, temperature=0.24):
        bs = z.shape[0]
        emb_size = args.stroke_emd_size

        start_of_stroke = -1 * torch.ones([bs, 1, emb_size]).cuda().float()
        start_of_pos = -1 * torch.ones([bs, 1, emb_size]).cuda().float()
        start_of_seq = torch.zeros([bs, 1, 1, 5]).cuda().float()
        start_of_seq[:, :, :, 2] = 1

        max_stroke_num = args.max_stroke_num
        max_stroke_len = args.max_stroke_len
        output_strokes = torch.zeros([bs, max_stroke_num, emb_size]).cuda().float()
        # output_strokes[:, 0] = start_of_stroke.squeeze(1)

        output_pos = torch.zeros([bs, max_stroke_num, 2]).cuda().float()

        output_seqs = torch.zeros([bs, max_stroke_num, max_stroke_len + 1, 5]).cuda().float()
        output_seqs[:, :, :, 4] = 1
        output_seqs[:, :, 0] = start_of_seq.squeeze(1)

        current_stroke = start_of_stroke
        current_pos = start_of_pos
        hidden_and_cell_1 = None
        hidden_and_cell_3 = None
        for stroke_num in range(max_stroke_num):
            pred_stroke, stopper_logits, pred_stopper, hidden_and_cell_1 = self.stroke_decoder(z, current_stroke, current_pos, hidden_and_cell_1=hidden_and_cell_1)

            idx_eos = self.get_pi_idx(random.random(), pred_stopper[0, 0].cpu().numpy(), temperature)
            eos = np.zeros(2)
            eos[idx_eos] = 1
            if eos[1] == 1:  # finish sketch drawing
                break

            output_strokes[0, stroke_num] = pred_stroke.squeeze(1)

            if stroke_num == 0:
                pred_pos = torch.tensor([0, 0]).cuda().float()
                absx = pred_pos[0]
                absy = pred_pos[1]
            else:
                current_pred_stroke_emb = torch.cat([current_stroke, pred_stroke], dim=1)
                pos_mu1, pos_mu2, pos_sigma1, pos_sigma2, pos_corr, hidden_and_cell_3 = self.pos_decoder(z, current_pred_stroke_emb, current_pos, hidden_and_cell_3=hidden_and_cell_3)
                next_x1, next_x2 = self.sample_gaussian_2d(pos_mu1[0, 0][0].cpu().numpy(),
                                                           pos_mu2[0, 0][0].cpu().numpy(),
                                                           pos_sigma1[0, 0][0].cpu().numpy(),
                                                           pos_sigma2[0, 0][0].cpu().numpy(),
                                                           pos_corr[0, 0][0].cpu().numpy(), np.sqrt(temperature))
                absx += next_x1
                absy += next_x2
                pred_pos = torch.tensor([absx, absy]).cuda().float()

            output_pos[0, stroke_num] = pred_pos

            current_stroke = pred_stroke
            current_pos = self.seq_encoder.pos2emb(torch.tensor([absx, absy]).cuda().float().reshape(1, 1, 2))  # compute embedding of the current position

            current_seq = start_of_seq
            hidden_and_cell_2 = None
            for stroke_len in range(max_stroke_len):
                pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits, hidden_and_cell_2 = self.seq_decoder(current_seq, z, pred_stroke,
                                                                                                          hidden_and_cell_2=hidden_and_cell_2)

                idx = self.get_pi_idx(random.random(), pi[0,0].cpu().numpy(), temperature)
                next_x1, next_x2 = self.sample_gaussian_2d(mu1[0,0][idx].cpu().numpy(), mu2[0,0][idx].cpu().numpy(),
                                                           sigma1[0,0][idx].cpu().numpy(), sigma2[0,0][idx].cpu().numpy(),
                                                           corr[0,0][idx].cpu().numpy(), np.sqrt(temperature))
                # generate stroke pen status
                idx_eos = self.get_pi_idx(random.random(), pen[0,0].cpu().numpy(), temperature)

                eos = np.zeros(3)
                eos[idx_eos] = 1

                output_seqs[0, stroke_num, stroke_len + 1, :] = torch.tensor([next_x1, next_x2, eos[0], eos[1], eos[2]]).cuda().float()

                current_seq = np.array([next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
                current_seq = torch.tensor(current_seq.reshape([1, 1, 1, 5])).cuda().float()

                if (eos[1] + eos[2]) == 1:  # finish stroke drawing
                    break

        return output_seqs, output_strokes, output_pos


    def eval(self, dataloader, category):
        self.seq_encoder.eval()
        self.stroke_encoder.eval()
        self.stroke_decoder.eval()
        self.seq_decoder.eval()
        self.pos_decoder.eval()

        count = 0
        seed = np.load('/data/datasets/quickdraw/random_seed.npy', allow_pickle=True)

        for batch in dataloader:
            seqs, seq_lens, padded_strokes, start_positions, stroke_lens, stroke_nums, _ = batch
            # seqs = seqs.cuda().float()
            # seq_lens = seq_lens.long()
            padded_strokes = padded_strokes.cuda().float()
            start_positions = start_positions.cuda().float()
            stroke_lens = stroke_lens.long()
            stroke_nums = stroke_nums.long()

            bs, num = padded_strokes.shape[0:2]
            stroke_stop_marker = torch.zeros([bs, num, 2])
            for i in range(bs):
                stroke_stop_marker[i, 0:stroke_nums[i], 0] = 1
                stroke_stop_marker[i, stroke_nums[i]:, 1] = 1

            with torch.no_grad():
                stroke_emb, pos_emb = self.seq_encoder(padded_strokes, start_positions, stroke_lens + 1, stroke_stop_marker[:, :, 0])
                seq_z = self.stroke_encoder(stroke_emb, pos_emb, stroke_nums)

                filepath = './sample/gt_%d_%d.npy' % (category, count)
                np.save(filepath, seq_z.cpu().numpy()[0])

                out_padded_seqs, out_strokes, out_pos = self.sample(seq_z, args)

                out_seqs = padded_5d_to_strokes_5d(out_padded_seqs, out_pos)
                filepath = './sample/seq_%d_%d.npy' % (category, count)
                np.save(filepath, out_seqs[0])

                temp = seq_5d_to_3d(out_seqs[0])
                if len(temp) < 2:
                    continue
                ########################
                temp[0, 0:2] = 0
                ########################
                _, graph, _, _, _, _ = make_graph_(temp, seed, seed_id=0, graph_num=20,
                                                   graph_picture_size=256, mask_prob=0.0, train=False)
                path = os.path.join("./sample/%d_%d.png" % (category, count))
                g0 = 255. - (graph[0] + 1) / 2. * 255.
                cv2.imwrite(path, np.tile(g0, [1, 1, 3]))

                fake_padded_strokes, fake_start_positions, fake_stroke_lens, fake_stroke_nums = translate(out_seqs, args)
                fake_padded_strokes = torch.tensor(fake_padded_strokes).cuda().float()
                fake_start_positions = torch.tensor(fake_start_positions).cuda().float()
                fake_stroke_lens = torch.tensor(fake_stroke_lens).long()
                fake_stroke_nums = torch.tensor(fake_stroke_nums).long()

                fake_stroke_stop_marker = torch.zeros([bs, num, 2])
                for i in range(bs):
                    fake_stroke_stop_marker[i, 0:fake_stroke_nums[i], 0] = 1
                    fake_stroke_stop_marker[i, fake_stroke_nums[i]:, 1] = 1

                fake_stroke_emb, fake_pos_emb = self.seq_encoder(fake_padded_strokes, fake_start_positions, fake_stroke_lens + 1, fake_stroke_stop_marker[:, :, 0])
                fake_seq_z = self.stroke_encoder(fake_stroke_emb, fake_pos_emb, fake_stroke_nums)

                filepath = './sample/fake_%d_%d.npy' % (category, count)
                np.save(filepath, fake_seq_z.cpu().numpy()[0])

            count += 1
            if count >= NUM_PER_CATEGORY:
                break

def sample(args):
    bs = 1

    args.epoch_load = EPOCH_LOAD

    model = Eval_Model(args)
    model.load(f'./sketch_model/seq_enc_epoch_{args.epoch_load}.pth',
               f'./sketch_model/stroke_enc_epoch_{args.epoch_load}.pth',
               f'./sketch_model/stroke_dec_epoch_{args.epoch_load}.pth',
               f'./sketch_model/seq_dec_epoch_{args.epoch_load}.pth',
               f'./sketch_model/pos_dec_epoch_{args.epoch_load}.pth')

    if not os.path.exists('./sample/'):
        os.makedirs('./sample/')

    for category in range(len(args.categories)):
        print(args.categories[category])
        testset = Dataset(args.data_dir, [args.categories[category]], args.max_stroke_num, args.max_stroke_len, mode="test")
        test_dataloader = DataLoader(testset, batch_size=bs, num_workers=16, shuffle=False)
        model.eval(test_dataloader, category)

    # Visualize latent codes
    for label in range(len(args.categories)):
        code_paths = glob.glob('./sample/fake_%d_*.npy' % label)  # Directory for generations
        code_paths = sort_paths(code_paths)

        img_paths = glob.glob('./sample/%d_*.png' % label)  # Directory for generations
        img_paths = sort_paths(img_paths)

        if label == 0:
            code = np.array(code_paths)
            img = np.array(img_paths)
        else:
            code = np.hstack((code, np.array(code_paths)))
            img = np.hstack((img, np.array(img_paths)))

    i = 0
    for path in code:
        temp = np.load(path)
        temp = np.expand_dims(temp, axis=0)
        # temp = np.reshape(np.load(path), [-1, model_params.enc_rnn_size])
        if i == 0:
            code_data = temp
        else:
            code_data = np.concatenate([code_data, temp], axis=0)
        i += 1

    tsne = TSNE(n_components=2).fit_transform(code_data)
    plt.scatter(tsne[:, 0], -tsne[:, 1], marker='.', c='black', s=1)
    plt.savefig('./cluster.png')

    img_data = []
    for path in img:
        img_data.append(Image.open(path).convert(mode='RGB'))
    grid_image_op = arrangeTsneByDist(tsne, img_data, width=4000, height=3000, max_dim=48)
    saveImage2Path(grid_image_op, './result.jpg')


if __name__ == "__main__":
    args = HParams()
    sample(args)

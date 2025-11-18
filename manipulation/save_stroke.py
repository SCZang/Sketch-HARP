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
BASE_DIR = "/workspace/0530-5cate-model/sample/stroke_save"

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

def visualize_strokes_by_index(padded_strokes, abs_start_positions, category, idx, stroke_num, base_dir):

    padded_strokes = padded_strokes.detach().cpu().squeeze(0).numpy()     
    abs_start_positions = abs_start_positions.detach().cpu().squeeze(0).numpy() 

    stroke_num_int = stroke_num.item()  
    max_stroke_num = min(padded_strokes.shape[0], stroke_num_int)

    os.makedirs(base_dir, exist_ok=True)

    for highlight_idx in range(max_stroke_num):
        plt.figure(figsize=(1.28, 1.28), dpi=100)

        for i in range(max_stroke_num):
            stroke = padded_strokes[i]
            start_pos = abs_start_positions[i][:2]  

            stroke = stroke[1:]  

            valid_idx = np.where(stroke[:, 4] == 0)[0]
            if len(valid_idx) <= 1:
                continue

            dx = stroke[valid_idx, 0]
            dy = stroke[valid_idx, 1]
            x = np.cumsum(dx) + start_pos[0]
            y = np.cumsum(dy) + start_pos[1]

            color = 'red' if i == highlight_idx else 'black'
            plt.plot(x, -y, linewidth=2, color=color)
            #if i == highlight_idx:
            #    plt.scatter(x[0], -y[0], color='red', s=10)

        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout(pad=0)

        filename = f"stroke_{highlight_idx:02d}.png"
        filepath = os.path.join(base_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.0)
        plt.close()

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

    def sample(self, z, args, temperature=0.24,inject_stroke_emb_dir=None,replace_dict=None): 
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

        s= 0
        absx, absy = 0.0, 0.0

        while inject_stroke_emb_dir and s in inject_stroke_emb_dir and "e" in inject_stroke_emb_dir[s]:
            inject_entry = inject_stroke_emb_dir[s]

            new_e = np.load(inject_entry["e"])
            new_e = torch.tensor(new_e).reshape(1, -1).cuda().float()
            pred_stroke = new_e.unsqueeze(1)

            _, _, _, hidden_and_cell_1 = self.stroke_decoder(z, current_stroke, current_pos, hidden_and_cell_1=hidden_and_cell_1)

            current_pred_stroke_emb = torch.cat([current_stroke, pred_stroke], dim=1)
            pos_mu1, pos_mu2, pos_sigma1, pos_sigma2, pos_corr, hidden_and_cell_3 = self.pos_decoder(
                z, current_pred_stroke_emb, current_pos, hidden_and_cell_3=hidden_and_cell_3
            )
            next_x1, next_x2 = self.sample_gaussian_2d(
                pos_mu1[0, 0][0].cpu().numpy(),
                pos_mu2[0, 0][0].cpu().numpy(),
                pos_sigma1[0, 0][0].cpu().numpy(),
                pos_sigma2[0, 0][0].cpu().numpy(),
                pos_corr[0, 0][0].cpu().numpy(),
                np.sqrt(temperature)
            )
            absx += next_x1
            absy += next_x2
            pred_pos = torch.tensor([absx, absy]).cuda().float()

            if "p" in inject_entry:
                pred_pos = inject_entry["p"]
                absx, absy = pred_pos[0].item(), pred_pos[1].item()
                output_pos[0, s] = pred_pos
            else:
                output_pos[0, s] = pred_pos

            output_strokes[0, s] = pred_stroke.squeeze(1)

            injected_seq = inject_entry["s"]
            if injected_seq.dim() == 2:
                injected_seq = injected_seq.unsqueeze(0)
            output_seqs[0, s, :injected_seq.shape[1], :] = injected_seq[0]

            current_stroke = pred_stroke
            current_pos = self.seq_encoder.pos2emb(torch.tensor([absx, absy]).cuda().float().reshape(1, 1, 2))
            s += 1
        
        for stroke_num in range(s, max_stroke_num):
            pred_stroke, stopper_logits, pred_stopper, hidden_and_cell_1 = self.stroke_decoder(z, current_stroke, current_pos, hidden_and_cell_1=hidden_and_cell_1)

          
            if replace_dict is not None and stroke_num in replace_dict:
                new_e = np.load(replace_dict[stroke_num])
                new_e = torch.tensor(new_e).reshape(1, -1).cuda().float()
                pred_stroke = new_e.unsqueeze(1)  

            idx_eos = self.get_pi_idx(random.random(), pred_stopper[0, 0].cpu().numpy(), temperature)
            eos = np.zeros(2)
            eos[idx_eos] = 1
            if eos[1] == 1: 
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

                if (eos[1] + eos[2]) == 1:  
                    break

        return output_seqs, output_strokes, output_pos

    def eval(self, dataloader, category,target_index=None):
        self.seq_encoder.eval()
        self.stroke_encoder.eval()
        self.stroke_decoder.eval()
        self.seq_decoder.eval()
        self.pos_decoder.eval()

        count = 0
        base_dir = BASE_DIR
        seed = np.load('/workspace/dataset/random_seed.npy', allow_pickle=True)

        for batch in dataloader:
            if target_index is not None and count != target_index:
                count += 1
                continue

            seqs, seq_lens, padded_strokes, start_positions, stroke_lens, stroke_nums, _ = batch
            
            sample_path = os.path.join(base_dir, f"sample_{category}_{count}")
            os.makedirs(sample_path, exist_ok=True)

            for i in range(padded_strokes.shape[0]):
                strokes_i = padded_strokes[i].detach().cpu().numpy()    
                pos_i = start_positions[i].detach().cpu().numpy()      
                num_strokes = int(stroke_nums[i])                                  
                for j in range(num_strokes):
                    s_j = strokes_i[j]  
                    p_j = pos_i[j]      
                    np.save(os.path.join(sample_path, f"s_{j:02d}.npy"), s_j)
                    np.save(os.path.join(sample_path, f"p_{j:02d}.npy"), p_j)
            visualize_strokes_by_index(padded_strokes, start_positions, category, count, stroke_nums, sample_path)

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
                save_stroke_emb = stroke_emb.squeeze(0).cpu().numpy()
                for i in range(min(save_stroke_emb.shape[0], stroke_nums)):
                    np.save(os.path.join(sample_path, f"z_{i:02d}.npy"), save_stroke_emb[i])
            count += 1
            if count >= NUM_PER_CATEGORY:
                break

def sample(args):
    bs = 1

    args.epoch_load = EPOCH_LOAD

    model = Eval_Model(args)
    model.load(f'/workspace/0530-5cate-model/model/seq_enc_epoch_{args.epoch_load}.pth',
               f'/workspace/0530-5cate-model/model/stroke_enc_epoch_{args.epoch_load}.pth',
               f'/workspace/0530-5cate-model/model/stroke_dec_epoch_{args.epoch_load}.pth',
               f'/workspace/0530-5cate-model/model/seq_dec_epoch_{args.epoch_load}.pth',
               f'/workspace/0530-5cate-model/model/pos_dec_epoch_{args.epoch_load}.pth')


    for category in range(len(args.categories)):
        print(args.categories[category])
        testset = Dataset(args.data_dir, [args.categories[category]], args.max_stroke_num, args.max_stroke_len, mode="test")
        test_dataloader = DataLoader(testset, batch_size=bs, num_workers=8, shuffle=False)
        model.eval(test_dataloader, category)


if __name__ == "__main__":
    args = HParams()
    sample(args)


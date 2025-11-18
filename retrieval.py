import torch
import numpy as np
import glob
import re
from train import Dataset, HParams

args = HParams()

original_z_code = []
generated_z_code = []

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

if __name__ == '__main__':

    for label in range(len(args.categories)):
        original_code_paths = glob.glob('./sample/gt_%d_*.npy' % label)  # Directory for generations
        original_code_paths = sort_paths(original_code_paths)

        generated_code_paths = glob.glob('./sample/fake_%d_*.npy' % label)  # Directory for generations
        generated_code_paths = sort_paths(generated_code_paths)

        if label == 0:
            original_code = np.array(original_code_paths)
            generated_code = np.array(generated_code_paths)
        else:
            original_code = np.hstack((original_code, np.array(original_code_paths)))
            generated_code = np.hstack((generated_code, np.array(generated_code_paths)))

    i = 0
    for path in original_code:
        temp = np.load(path)
        temp = np.expand_dims(temp, axis=0)
        if i == 0:
            ori_code_data = temp
        else:
            ori_code_data = np.concatenate([ori_code_data, temp], axis=0)
        i += 1

    i = 0
    for path in generated_code:
        temp = np.load(path)
        temp = np.expand_dims(temp, axis=0)
        if i == 0:
            fake_code_data = temp
        else:
            fake_code_data = np.concatenate([fake_code_data, temp], axis=0)
        i += 1

    ori_code_data = torch.from_numpy(ori_code_data).cuda()
    fake_code_data = torch.from_numpy(fake_code_data).cuda()

    top_1 = 0
    top_10 = 0
    top_50 = 0

    for i in range(len(ori_code_data)):
        dist = torch.norm(ori_code_data[i].view(-1, ori_code_data.shape[-1])[:, None] - fake_code_data, 2, 2)
        sorted_index = torch.argsort(dist).detach().cpu()
        if i == sorted_index[0, 0]:
            top_1 += 1
        if i in sorted_index[0, :9]:
            top_10 += 1
        if i in sorted_index[0, :49]:
            top_50 += 1


    print('top 1:', top_1 / len(ori_code_data))
    print('top 10:', top_10 / len(ori_code_data))
    print('top 50:', top_50 / len(ori_code_data))

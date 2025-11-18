# Sketch-HARP: Generating Sketches in a Hierarchical Auto-Regressive Process for Flexible Sketch Drawing Manipulation at Stroke-Level

Sketch-HARP is a framework to manipulate free-hand sketch drawing by a hierarchical auto-regressive process, shown in the overview below. Instead of generating an entire sketch at once, each stroke in a sketch is generated in a three-staged hierarchy: 1) predicting a stroke embedding to represent which stroke is going to be drawn, and 2) anchoring the predicted stroke on the canvas, and 3) translating the embedding to a sequence of drawing actions to form the full sketch. Thus, it is flexible to manipulate stroke-level sketch drawing at any time during generation by adjusting the exposed editable stroke embeddings. 

This is the official released codes, and its corresponding article was accepted by **AAAI 2026** (an early access version is available at [arxiv](https://arxiv.org/abs/2511.07889)).

<img src="https://github.com/SCZang/Sketch-HARP/blob/4293a9492024afb51a068eb22a2a0dad6a5a934f/assets/overview.jpg" width="800" alt="overview"/>

# Training a Sketch-HARP

## Datasets

We use [QuickDraw dataset](https://quickdraw.withgoogle.com/data) to train our Sketch-HARP.

## Required environments

See `requirements.txt`.

## Training

The training settings can be found at the class `HParams` in `train.py`.

```
data_dir = "/workspace/dataset"      # dataset directory
categories = ["bee", "bus", "flower", "giraffe", "pig"]  # sketch categories selected for training
enc_rnn_size_1 = 512                 # hidden state size of the stroke encoder
enc_rnn_size_2 = 512                 # hidden state size of the sketch encoder
stroke_emd_size = 128                # stroke embedding size
dec_rnn_size_1 = 1024                # hidden state size of the stroke decoder
dec_rnn_size_2 = 1024                # hidden state size of the position decoder
dec_rnn_size_3 = 1024                # hidden state size of the sequence decoder
max_stroke_num = 25                  # maximum number of strokes allowed per sketch
max_stroke_len = 32                  # maximum number of drawing actions allowed per stroke
zdim = 128                           # latent code size
lr = 0.001                           # initialized learning rate
bs = 128                             # mini-batch size
num_epochs = 100                     # max number of training epochs
epoch_load = 0                       # load a pre-trained model from this epoch (0 for training from scratch)
```

After setting, you can simply run
```
python train.py
```
to starting network training.

## Generating sketches

Generating sketches by using a pre-trained model. We offer two [pretrained models](https://pan.baidu.com/s/1-NkCB4ypdReIX4SzI2SxWA?pwd=hb97) for two datasets, respectively.

```
python sample.py
```

In `sample.py`, `EPOCH_LOAD` in *line 19* indicates the number of epochs of the pre-trained model, and `NUM_PER_CATEGORY` in *line 20* indicates the number of generated items per category.

## Evaluation

We use **CLIP score**, **LPIPS** and **FID** to evaluate the quality of generated sketches, with the calculating codes borrowed from [SketchEdit](https://github.com/CMACH508/SketchEdit). You are able to save the generated sketches in `./results/` and the ground truth sketches in `./GroundTruth/`, respectively, and evaluate the performances.

```
cd evaluations
python CLIP_score.py ../results/ ../GroundTruth/ --real_flag img --fake_flag img --device cuda
python fid_score.py ../results/ ../GroundTruth/ --gpu 0
python lpips_score.py --path1 ../results/ --path2 ../GroundTruth/
```

Besides, the metrics, **Rec** and **Ret**, are used to testify whether a method learns accurate and robust sketch representations. For calculating **Rec**, you need to train a [Sketch-a-net](https://arxiv.org/pdf/1501.07873.pdf) for each dataset as the classifier. And for **Ret**, you can run `retrieval.py`  to obtain it with the generated sketches (2500 sketches per category). Details see [link](https://github.com/CMACH508/SP-gra2seq).

# Manipulating sketch drawing at stroke-level

##  Stroke preparation

Before applying sketch manipulation, we prepare stroke (embeddings) from sketches in test set, and these strokes would be used as elements in sketch manipulation. You are able to run
```
cd manipulation
python save_stroke.py
```
to save all the stroke embeddings (as well as their corresponding positions and sequence of drawing actions) captured from sketches into `./BASE_DIR/` (*line 20* in `save_stroke.py`).

## Replacement

<img src="/assets/replacement.png" width="600" alt="replacement"/>

```
TARGET_CATEGORY = 1                                  # the target sketch category
TARGET_SAMPLE_IDX = 2                                # the target sketch sample
TARGET_STROKE_IDX = 3                                # the target stroke to be manipulated
SOURCE_EMBEDDING_PATH = "./source_stroke_emb.npy"    # the selected source stroke embedding
```

Replacing sketch strokes has three steps:
1) Select the target sketch (to be manipulated) by `TARGET_CATEGORY` and `TARGET_SAMPLE_IDX`, and pick the target stroke (to be replaced) by `TARGET_STROKE_IDX`.
2) Select the source stroke (providing the manipulating source stroke embedding(s)) by `SOURCE_EMBEDDING_PATH`.
3) Run 
```
python replace.py
```
and you will find the manipulated sketches stored in './sample/'.


## Expansion

<img src="/assets/expansion.png" width="600" alt="expansion"/>

```
TARGET_CATEGORY = 3                                # the target sketch category
TARGET_SAMPLE_IDX = 36                             # the target sketch sample
INJECT_STROKE_EMB_DIR = {
    0: {                                           # the first source stroke to be injected into sketch drawing
        "e": "/stroke_save/sample_8_10/z_00.npy",  # the selcted stroke embedding
        "p": torch.tensor([0.0, 0.0]).cuda(),      # its starting position
        "s": /stroke_save/sample_8_10/s_00.npy"    # its corresponding sequence of drawing actions 
    },
    1: {                                           # the second source stroke to be injected into sketch drawing
        "e": "/stroke_save/sample_9_20/z_01.npy",
        "p": torch.tensor([0.0, 0.0]).cuda(),
        "s": /stroke_save/sample_9_20/s_01.npy"
    },
}
```

Expanding sketch strokes has three steps:
1) Select the target sketch (to be manipulated) by `TARGET_CATEGORY` and `TARGET_SAMPLE_IDX`.
2) Select the source stroke(s) (to be injected into sketch drawing) by `INJECT_STROKE_EMB_DIR`.
3) Run 
```
python expand.py
```
and you will find the manipulated sketches stored in './sample/'.

## Erasion

<img src="/assets/erasion.png" width="600" alt="erasion"/>

```
TARGET_CATEGORY = 3       # The target sketch (category) to be manipulated
TARGET_SAMPLE_IDX = 36    # The target sketch sample to be manipulated
DELETE_INDICES = [0, 3]   # indexes of strokes to be erased
```
Erasing sketch strokes has three steps:
1) Select the target sketch (to be manipulated) by `TARGET_CATEGORY` and `TARGET_SAMPLE_IDX`.
2) Select the target stroke(s) (to be erased) by `DELETE_INDICES`.
3) Run 
```
python erase.py
```
and you will find the manipulated sketches stored in './sample/'.

# Citation
If you find this project useful for academic purposes, please cite it as:

```
@article{zang2025generating,
  title={Generating Sketches in a Hierarchical Auto-Regressive Process for Flexible Sketch Drawing Manipulation at Stroke-Level},
  author={Zang, Sicong and Gao, Shuhui and Fang, Zhijun},
  journal={arXiv preprint arXiv:2511.07889},
  year={2025}
}
```









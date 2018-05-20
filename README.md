# caption-stylenet_tensorflow
# StyleNet: Generating Attractive Visual Captions with Styles

StyleNet is a novel framework to address the task of generating attractive captions for images and videos with different styles. Authors proposed a model, which is based on using
factorized LSTM.


## Description
- Author: Chuang Gan, Zhe Gan, Xiaodong He, Jianfeng Gao, Li Deng
- Published in: Computer Vision and Pattern Recognition (CVPR), 2017
- URL:  https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/Generating-Attractive-Visual-Captions-with-Styles.pdf

## Requires
- Tensorflow 1.8.0 (lower versions supposed to work too)
- python 3
- numpy
- tqdm
- opencv-python

## Usage
- first follow preprocessing notebook for image captions preprocessing
- place preprocessed captions into ./pickles folder
- download VGG16 weights from: https://yadi.sk/d/V6Rfzfei3TdKCH
- place downloaded weights to ./utils folder
- don't forget to download data and set the path int the parameters -(https://zhegan27.github.io/Papers/FlickrStyle_v0.9.zip)- flickr_style7k
- to train launch:
```
 python main.py --gpu <YOUR_GPU>
```
- for other parameters look at source code or just:
```
  python main.py -h
```
- to generate:
```
  python main.py --gpu <YOUR_GPU> <Other parameters you used> --mode inference
  --gen_label <romantic, humorous, actual> --gen_name <default 00>
```

## TODO
- change VGG16 to ResNet
- try to get better results (results are worse than reported in paper)
- add generation for arbitary photos

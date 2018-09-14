# StyleNet: Generating Attractive Visual Captions with Styles	# ResNet in TensorFlow

 StyleNet is a novel framework to address the task of generating attractive captions for images and videos with different styles. Authors proposed a model, which is based on using	Deep residual networks, or ResNets for short, provided the breakthrough idea of identity mappings in order to enable training of very deep convolutional neural networks. This folder contains an implementation of ResNet for the ImageNet dataset written in TensorFlow.
factorized LSTM.	
 See the following papers for more background:

 ## Description	[1] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

- Author: Chuang Gan, Zhe Gan, Xiaodong He, Jianfeng Gao, Li Deng	
- Published in: Computer Vision and Pattern Recognition (CVPR), 2017	[2] [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
- URL:  https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/Generating-Attractive-Visual-Captions-with-Styles.pdf	
 In code, v1 refers to the ResNet defined in [1] but where a stride 2 is used on

## Requires	the 3x3 conv rather than the first 1x1 in the bottleneck. This change results

- Tensorflow 1.8.0 (lower versions supposed to work too)	in higher and more stable accuracy with less epochs than the original v1 and has
- python 3	shown to scale to higher batch sizes with minimal degradation in accuracy.
- numpy	There is no originating paper and the first mention we are aware of was in the
- tqdm	[torch version of ResNetv1](https://github.com/facebook/fb.resnet.torch). Most
- opencv-python	popular v1 implementations are this implementation which we call ResNetv1.5. In
 testing we found v1.5 requires ~12% more compute to train and has 6% reduced

## Usage	throughput for inference compared to ResNetv1. Comparing the v1 model to the
- first follow preprocessing notebook for image captions preprocessing	v1.5 model, which has happened in blog posts, is an apples-to-oranges
- place preprocessed captions into ./pickles folder	comparison especially in regards to hardware or platform performance. CIFAR-10
- download VGG16 weights from: https://yadi.sk/d/V6Rfzfei3TdKCH	ResNet does not use the bottleneck and is not impacted by these nuances.
- place downloaded weights to ./utils folder	
- don't forget to download data and set the path int the parameters -(https://zhegan27.github.io/Papers/FlickrStyle_v0.9.zip)- flickr_style7k	v2 refers to [2]. The principle difference between the two versions is that v1
- to train launch:
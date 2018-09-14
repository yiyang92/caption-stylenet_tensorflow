import argparse
import os


# parse arguments, return dictionary
def parse_args():
    parser = argparse.ArgumentParser(description="Specify training parameters,"
                                     "all parameters also can be "
                                     "directly specify in the "
                                     "Parameters class")
    parser.add_argument('--learning_rate', default=0.0002,
                        help='learning rate', type=float)
    parser.add_argument('--embed_dim', default=300,
                        help='embedding size', type=int)
    parser.add_argument('--keep_words', default=1,
                        help='minimum word occurence', type=int)
    parser.add_argument('--lstm_hidden', default=512,
                        help='lstm hidden state size', type=int)
    parser.add_argument('--num_captions', default=5,
                        help='Num captions feeding every batch (cm_model)',
                        type=int)
    parser.add_argument('--restore', help='whether restore',
                        action="store_true")
    parser.add_argument('--lstm_clip_norm', default=5.0,
                        help='whether to clip lstm gradients', type=float)
    parser.add_argument('--gpu', default='0', help="specify GPU number")
    def_img_dir = "/home/luoyy16/datasets-large/flickr30k-images/"
    parser.add_argument('--image_dir', default=def_img_dir,
                        help="flickr30k directory")
    parser.add_argument('--epochs', default=50,
                        help="number of training epochs", type=int)
    parser.add_argument('--batch_size', default=64,
                        help="Batch size", type=int)
    parser.add_argument('--batch_size_lm', default=96,
                        help="Batch size", type=int)
    parser.add_argument('--temperature', default=0.6, type=float,
                        help="set temperature parameter for generation")
    parser.add_argument('--gen_name', default='00',
                        help="prefix of generated json nam")
    parser.add_argument('--lstm_drop', default=1.0,
                        help="lstm dropout keep rate", type=float)
    parser.add_argument('--sample_gen', default='beam_search',
                        help="greedy, sample, beam_search")
    parser.add_argument('--checkpoint', default='00',
                        help="specify checkpoint name, default=last_run")
    parser.add_argument('--write_summary',
                        help="whether to write summary for tensorboard",
                        action="store_true")
    parser.add_argument('--mode', default='training',
                        choices=['training', 'inference'],
                        help="specify training or inference")
    parser.add_argument('--gen_label', default='actual',
                        choices=['actual', 'humorous', 'romantic'],
                        help="generation label")
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD', 'Momentum'],
                        help="specify optimizer")
    parser.add_argument('--beam_size', default=5,
                        help="beam size (default:5)", type=int)
    parser.add_argument('--gen_set', default='val',
                        choices=['val', 'test'],
                        help="test time caption generation set")
    parser.add_argument('--keep_cp', default=1,
                        help="max checkpoints to keep", type=int)
    parser.add_argument('--gen_max', default=50,
                        help="max caption length", type=int)
    parser.add_argument('--tr_style', default='romantic',
                        choices=['both', 'humorous', 'romantic'],
                        help="at this time train romantic, humorous or both",
                        type=str)
    parser.add_argument(
        "--img_embed", 
        default="vgg", 
        choices=["resnet", "vgg"])
    parser.add_argument(
       "--weights",
       default="./utils/resnet_v1_imagenet/model.ckpt-257706",
       help="Weights for vgg(npz file) or Resnet checkpoint"
    )


    args = parser.parse_args()
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return vars(args)

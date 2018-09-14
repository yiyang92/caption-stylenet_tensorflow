import pickle
import random
import os
import argparse
import json

import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--stc_dir", 
    help="Stylized caption directory",
    default="/home/luoyy16/datasets-large/FlickrStyle_v0.9/")
parser.add_argument(
    "--f30_cap_dir",
    help="flickr30k captions",
    default="/home/luoyy16/datasets-large/Flickr30kEntities/Sentences/")
parser.add_argument(
    "--f30_im_dir",
    help="flickr30k images",
    default="/home/luoyy16/datasets-large/flickr30k-images"
)
parser.add_argument(
    "--gen_an",
    help="generate test time annotations",
    default=False,
    action="store_true"
)
parser.add_argument(
    "--pickles_out",
    help="pickles (training captions) output dir",
    default="./pickles"
)
parser.add_argument(
    "--split",
    help="Factual captions split, random or karpathy",
    choices=["karp", "rand"],
    default="rand"
)
parser.add_argument(
    "--karp_path",
    default="/home/luoyy16/datasets-large/caption_datasets/dataset_flickr30k.json"
)
args = parser.parse_args()
# file directories
ST_CAP_DIR = args.stc_dir
F30_CAP_DIR = args.f30_cap_dir
F30_IM_DIR = args.f30_im_dir
# tags
start_tag = '<BOS>'
end_tag = '<EOS>'

def style_caps(pfn, capfn):
    with open(pfn, 'rb') as rf:
        cap_fn = pickle.load(rf)
    with open(capfn, 'rb') as rf:
        cap = []
        ctr = 0
        for line in rf:
            try:
                line = line.decode('utf-8')[:-1]
                ctr+=1
            except: 
                line = line.decode('utf-8', 'replace')[:-1]
                print("line {} contains weird characters".format(ctr+1))
                print(line)
            if len(line) < 4:
                # for some weird line
                continue
            cap.append(line.lower().rstrip())
    return cap_fn, cap

hum_cap_fn, hum_cap = style_caps(
    os.path.join(ST_CAP_DIR, 'humor/train.p'),
    os.path.join(ST_CAP_DIR, 'humor/funny_train.txt'))
rom_cap_fn, rom_cap = style_caps(
    os.path.join(ST_CAP_DIR, 'romantic/train.p'),
    os.path.join(ST_CAP_DIR, 'romantic/romantic_train.txt'))

print(
    "File names lists are equal in order: {}".format(hum_cap_fn == rom_cap_fn))

def get_act_caps(cap_path):
    cap_files = list(glob(cap_path + '*.txt'))
    cap_dict = {}
    for file in tqdm(cap_files):
        caps = []
        with open(file, 'r') as rf:
            lines = rf.readlines()
        for line in lines:
            line = line.strip().lower().split(' ')
            p_line = []
            for wd in line:
                wd_s = wd.split(']')
                if '[' in wd:
                    continue
                if len(wd_s) > 1:
                    wd_s = [wd_s[0]]
                p_line.extend(wd_s)
            caps.append(p_line)
        im_name = file.split('/')[-1].split('.')[0] + '.jpg'
        cap_dict[im_name] = caps
    return cap_dict

def get_karp_spit():
    """Karpathy split, returns dict f_name: [captions]"""
    print("WARNING: Karpathy split not working peopwrly now, use random split")
    with open(args.karp_path, "r") as rf:
        carp_split = json.load(rf)
    if not carp_split["dataset"] == "flickr30k":
        raise ValueError("Provided not f30k split")
    capt_dict = {}
    capt_dict_set = {}
    for c_d in carp_split["images"]:
        sent_list = [sent["tokens"] for sent in c_d["sentences"]]
        capt_dict[c_d["filename"]] = sent_list
        capt_dict_set[c_d["filename"]] = c_d["split"]
    return capt_dict, capt_dict_set

if args.split == "rand":
    # from original data
    cap_dict_30k = get_act_caps(F30_CAP_DIR)
elif args.split == "karp":
    cap_dict_30k, capt_dict_set = get_karp_spit()
else:
    ValueError("Can use rand (random) or karp(Karpathy) split")

imn30k = os.listdir(F30_IM_DIR)
cap_dict = {}
imn30kset = set(imn30k)
for i in range(len(rom_cap_fn)):
    imn = rom_cap_fn[i]
    imn = imn.split('_')[0] + '.jpg'
    cap_dict[imn] = {'romantic': [[start_tag] + rom_cap[i].split(' ') + [end_tag]], 
                     'humorous': [[start_tag] + hum_cap[i].split(' ') + [end_tag]], 
                     'actual': ''}
                    
NUM_TRAIN = int(len(rom_cap_fn) * 0.85)
NUM_VAL = int(len(rom_cap_fn) * 0.05)
NUM_TEST = int(len(rom_cap_fn) * 0.1)
print("labelled images split: train: {} test: {}".format(NUM_TRAIN, NUM_TEST))

def form_dict(orig_dict, keys):
    dest_dict = {}
    for key in keys:
        dest_dict[key] = orig_dict[key]
    return dest_dict

def split_labelled(cap_dict):
    # split into val and test
    keys = list(cap_dict.keys())
    keys_perm = np.random.permutation(keys)
    keys_tr = keys_perm[:NUM_TRAIN]
    keys_vl = keys_perm[NUM_TRAIN: (NUM_TRAIN + NUM_VAL)]
    keys_ts = keys_perm[(NUM_TRAIN + NUM_VAL):]
    keys_ts = keys_perm[NUM_TRAIN:]
    cap_tr = form_dict(cap_dict, keys_tr)
    cap_vl = form_dict(cap_dict, keys_vl)
    cap_ts = form_dict(cap_dict, keys_ts)
    return cap_tr, cap_vl, cap_ts

cap_dict, cap_lval, cap_ltest = split_labelled(cap_dict)
cap_dict_l = cap_dict.copy()
# add actual captions， flickr8k-subset of flickr30k
ctr1, ctr2 = 0, 0
for imn in imn30k:
    if imn == 'readme.txt':
        continue
    try:
        if imn in cap_dict_30k:
            cap_dict[imn].update(
                {'actual': [[start_tag] + cap + [end_tag] for cap in cap_dict_30k[imn]]})
        else:
            cap_dict.pop(imn, None)
            ctr1 += 1
    except:
        cap_dict[imn] = {
            'actual': [[start_tag] + cap + [end_tag] for cap in cap_dict_30k[imn]]}
    try:
        if imn in cap_dict_30k:
            cap_lval[imn].update(
                {'actual': [[start_tag] + cap + [end_tag] for cap in cap_dict_30k[imn]]})
        else:
            cap_lval.pop(imn, None)
            ctr1 += 1
    except:
        pass
    try:
        if imn in cap_dict_30k:
            cap_ltest[imn].update(
                {'actual': [[start_tag] + cap + [end_tag] for cap in cap_dict_30k[imn]]})
        else:
            cap_ltest.pop(imn, None)
    except:
        # As we want have stylized + factual full sets
        ctr2 += 1
        pass

print("Not included: ", ctr1, " Included: ", ctr2)
pickles_dir = args.pickles_out
# save to pickles
if not os.path.exists(pickles_dir):
    os.mkdir(pickles_dir)

# labelled + unlabelled captions
with open(os.path.join(pickles_dir, 'captions_tr.pkl'), 'wb') as wf:
    pickle.dump(file=wf, obj=cap_dict)
# only labelled
with open(os.path.join(pickles_dir, 'captions_ltr.pkl'), 'wb') as wf:
    pickle.dump(file=wf, obj=cap_dict_l)
# labelled val captions
with open('./pickles/captions_val.pkl', 'wb') as wf:
    pickle.dump(file=wf, obj=cap_lval)
# labelled test captions
with open(os.path.join(pickles_dir, 'captions_test.pkl'), 'wb') as wf:
    pickle.dump(file=wf, obj=cap_ltest)

# dataset overview
print(
    "Training set size: {}\nLabelled Training set size: {}\nValidation set size: {}\nTest set size: {}".format(
    len(cap_dict.keys()), len(cap_dict_l.keys()),len(cap_lval.keys()), len(cap_ltest.keys())))

# format: {'caption': 'A bicycle replica with a clock as the front wheel.',
#  'id': 37,
#  'image_id': 203564}
# use json dump for list of dictionaries


def prepare_eval(caption_dict, label):
    # annotations
    eval_d_list = []
    img_info = []
    for imid in caption_dict.keys():
        ev_dict = {'image_id': int(imid.split('.')[0]),
                   'caption': ' '.join(caption_dict[imid][label][0][1:-1]),
                   'id': int(imid.split('.')[0])}
        im_dict = {'id' : int(imid.split('.')[0]),
                   'file_name': imid}
        eval_d_list.append(ev_dict)
        img_info.append(im_dict)
    return {'annotations': eval_d_list, 'images': img_info}

if args.gen_an:
    if not os.path.exists('./annotations'):
        os.makedirs('./annotations')

    def dump_to_json(obj, f_name):
        with open('./annotations/' + f_name, 'w') as wf:
            json.dump(obj, wf)
    # test
    dump_to_json(prepare_eval(cap_ltest, 'actual'), 'test_act.json')
    dump_to_json(prepare_eval(cap_ltest, 'romantic'), 'test_rom.json')
    dump_to_json(prepare_eval(cap_ltest, 'humorous'), 'test_hum.json')
    # val
    dump_to_json(prepare_eval(cap_lval, 'actual'), 'val_act.json')
    dump_to_json(prepare_eval(cap_lval, 'romantic'), 'val_rom.json')
    dump_to_json(prepare_eval(cap_lval, 'humorous'), 'val_hum.json')
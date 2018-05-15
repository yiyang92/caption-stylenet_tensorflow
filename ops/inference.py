# inference op
import json
import os
import numpy as np


def inference(params, decoder, data, saver, sess):
    print("Restoring from checkpoint")
    saver.restore(sess, "./checkpoints/{}.ckpt".format(params['checkpoint']))
    # validation set
    captions_gen = []
    print("Generating captions for {} file".format(params['gen_set']))
    label = params['gen_label']
    for captions, lengths, image_f, names in data.get_batch(100,
                                                            set=params['gen_set'],
                                                            get_names=True,
                                                            label=label):
        if params['sample_gen'] == 'beam_search':
            pass
            # sent = decoder.beam_search(sess, names, image_f,
            #                            image_ph, labels,
            #                            ground_truth=captions[1],
            #                            beam_size=params.beam_size)
        else:
            sent, _ = decoder.online_inference(sess, names, image_f,
                                               label, ground_truth=captions[1])
        captions_gen += sent
    print("Generated {} captions".format(len(captions_gen)))
    gen_file = "./val_{}.json".format(params['gen_name'])
    if os.path.exists(gen_file):
        print("Exists {}, delete it".format(gen_file))
        os.remove(gen_file)
        print(os.listdir('.'))
    with open(gen_file, 'w') as wj:
        print("saving val json file into ", gen_file)
        json.dump(captions_gen, wj)

from __future__ import absolute_import
from __future__ import print_function
import collections
import glob
import os
import re
import sys
import numpy as np

import toolkits
import preprocess

import pdb
# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'pretrained/weights.h5', type=str)
parser.add_argument('--data_path', default='4persons', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--max_utt', default=100, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

global args
args = parser.parse_args()

MORE_TTS_RAW_PATH = '/nas0/poodle/speech_dataset/uncategorized/more-TTS-Jun2019/'
MORE_TTS_SPEAKERS = [
    'ko-KR-Wavenet-A_google',
    'ko-KR-Wavenet-B_google',
    'ko-KR-Wavenet-C_google',
    'ko-KR-Wavenet-D_google',
    'MAN_DIALOG_BRIGHT_kakao',
    'MAN_READ_CALM_kakao',
    'WOMAN_DIALOG_BRIGHT_kakao',
    'WOMAN_READ_CALM_kakao']
MORE_TTS_FILENAME_PATTERN = r'(ytn_)?(?P<utt_id>\d{5})_(?P<spk_id>[\w\-_]+).wav'
MORE_TTS_FILENAME_PATTERN_STR = '{utt_id:05d}_{spk_id}.wav'

MAX_UTT = 1000

def _prepare_data(max_num=100):
    utterences = collections.defaultdict(list)

    for spk_id in MORE_TTS_SPEAKERS:
        for i in range(max_num):
            filename = MORE_TTS_FILENAME_PATTERN_STR.format(utt_id=i, spk_id=spk_id)
            filepath = os.path.join(MORE_TTS_RAW_PATH, filename)
            yield filepath, spk_id

def _prepare_model(params=None):
    assert params
    import model
    # ==================================
    #       Get Train/Val.
    # ==================================

    # total_list = [os.path.join(args.data_path, file) for file in os.listdir(args.data_path)]
    # unique_list = np.unique(total_list)

    # ==================================
    #       Get Model
    # ==================================
    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)

    # ==> load pre-trained model ???
    if args.resume:
        # ==> get real_model from arguments input,
        # load the model if the imag_model == real_model.
        if os.path.isfile(args.resume):
            network_eval.load_weights(os.path.join(args.resume), by_name=True)
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            raise IOError("==> no checkpoint found at '{}'".format(args.resume))
    else:
        raise IOError('==> please type in the model to load')

    print('==> start testing.')

    return network_eval


def main():
    # gpu configuration
    toolkits.initialize_GPU(args)

    # construct the data generator.
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'min_slice': 720,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }
    # Prepare model
    network_eval = _prepare_model(params)

    # The feature extraction process has to be done sample-by-sample,
    # because each sample is of different lengths.
    feats = []
    targets = []
    for i, (ID, spk_id) in enumerate(_prepare_data(MAX_UTT)):
        print(i, ID)
        specs = preprocess.load_data(ID, split=False, win_length=params['win_length'], sr=params['sampling_rate'],
                             hop_length=params['hop_length'], n_fft=params['nfft'],
                             min_slice=params['min_slice'])
        specs = np.expand_dims(np.expand_dims(specs[0], 0), -1)

        v = network_eval.predict(specs)
        feats += [v]
        targets.append(spk_id)

    feats = np.array(feats)[:,0,:]
    targets = np.array(targets)
    np.savez(f'embeddings_{MAX_UTT}', feats=feats, targets=targets)
    preprocess.similar(feats)


if __name__ == "__main__":
    main()


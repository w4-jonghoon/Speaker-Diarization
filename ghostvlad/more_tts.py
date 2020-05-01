import glob
import os
import re

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

def prepare_data_all():
    files = glob.iglob(os.path.join(MORE_TTS_RAW_PATH, '*.wav'))
    filename_pattern = re.compile(MORE_TTS_FILENAME_PATTERN)

    path_spk_list = []
    for filename in files:
        searched = filename_pattern.search(filename)
        if searched is None:
            continue

        filepath = os.path.join(MORE_TTS_RAW_PATH, filename)
        path_spk_list.append((filepath, searched.group('spk_id')))
    return path_spk_list


if __name__ == '__main__':
    _prepare_data_all()

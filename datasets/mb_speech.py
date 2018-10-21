"""Data loader for the Mongolian Bible dataset."""
import os
import codecs
import numpy as np

from torch.utils.data import Dataset

vocab = "PE абвгдеёжзийклмноөпрстуүфхцчшъыьэюя-.,!?"  # P: Padding, E: EOS.
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}


def text_normalize(text):
    text = text.lower()
    # text = text.replace(",", "'")
    # text = text.replace("!", "?")
    for c in "-—:":
        text = text.replace(c, "-")
    for c in "()\"«»“”'":
        text = text.replace(c, ",")
    return text


def read_metadata(metadata_file):
    fnames, text_lengths, texts = [], [], []
    transcript = os.path.join(metadata_file)
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    for line in lines:
        fname, _, text = line.strip().split("|")

        fnames.append(fname)

        text = text_normalize(text) + "E"  # E: EOS
        text = [char2idx[char] for char in text]
        text_lengths.append(len(text))
        texts.append(np.array(text, np.long))

    return fnames, text_lengths, texts


def get_test_data(sentences, max_n):
    normalized_sentences = [text_normalize(line).strip() + "E" for line in sentences]  # text normalization, E: EOS
    texts = np.zeros((len(normalized_sentences), max_n + 1), np.long)
    for i, sent in enumerate(normalized_sentences):
        texts[i, :len(sent)] = [char2idx[char] for char in sent]
    return texts


class MBSpeech(Dataset):
    def __init__(self, keys, dir_name='MBSpeech-1.0'):
        self.keys = keys
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dir_name)
        self.fnames, self.text_lengths, self.texts = read_metadata(os.path.join(self.path, 'metadata.csv'))

    def slice(self, start, end):
        self.fnames = self.fnames[start:end]
        self.text_lengths = self.text_lengths[start:end]
        self.texts = self.texts[start:end]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        data = {}
        if 'texts' in self.keys:
            data['texts'] = self.texts[index]
        if 'mels' in self.keys:
            # (39, 80)
            data['mels'] = np.load(os.path.join(self.path, 'mels', "%s.npy" % self.fnames[index]))
        if 'mags' in self.keys:
            # (39, 80)
            data['mags'] = np.load(os.path.join(self.path, 'mags', "%s.npy" % self.fnames[index]))
        if 'mel_gates' in self.keys:
            data['mel_gates'] = np.ones(data['mels'].shape[0], dtype=np.int)  # TODO: because pre processing!
        if 'mag_gates' in self.keys:
            data['mag_gates'] = np.ones(data['mags'].shape[0], dtype=np.int)  # TODO: because pre processing!
        return data

#
# simple method to convert mongolian numbers to text, copied from somewhere
#


def number2word(number):
    digit_len = len(number)
    digit_name = {1: '', 2: 'мянга', 3: 'сая', 4: 'тэрбум', 5: 'их наяд', 6: 'тунамал'}

    if digit_len == 1:
        return _last_digit_2_str(number)
    if digit_len == 2:
        return _2_digits_2_str(number)
    if digit_len == 3:
        return _3_digits_to_str(number)
    if digit_len < 7:
        return _3_digits_to_str(number[:-3], False) + ' ' + digit_name[2] + ' ' + _3_digits_to_str(number[-3:])

    digitgroup = [number[0 if i - 3 < 0 else i - 3:i] for i in reversed(range(len(number), 0, -3))]
    count = len(digitgroup)
    i = 0
    result = ''
    while i < count - 1:
        result += ' ' + (_3_digits_to_str(digitgroup[i], False) + ' ' + digit_name[count - i])
        i += 1
    return result.strip() + ' ' + _3_digits_to_str(digitgroup[-1])


def _1_digit_2_str(digit):
    return {'0': '', '1': 'нэгэн', '2': 'хоёр', '3': 'гурван', '4': 'дөрвөн', '5': 'таван', '6': 'зургаан',
            '7': 'долоон', '8': 'найман', '9': 'есөн'}[digit]


def _last_digit_2_str(digit):
    return {'0': 'тэг', '1': 'нэг', '2': 'хоёр', '3': 'гурав', '4': 'дөрөв', '5': 'тав', '6': 'зургаа', '7': 'долоо',
            '8': 'найм', '9': 'ес'}[digit]


def _2_digits_2_str(digit, is_fina=True):
    word2 = {'0': '', '1': 'арван', '2': 'хорин', '3': 'гучин', '4': 'дөчин', '5': 'тавин', '6': 'жаран', '7': 'далан',
             '8': 'наян', '9': 'ерэн'}
    word2fina = {'10': 'арав', '20': 'хорь', '30': 'гуч', '40': 'дөч', '50': 'тавь', '60': 'жар', '70': 'дал',
                 '80': 'ная', '90': 'ер'}
    if digit[1] == '0':
        return word2fina[digit] if is_fina else word2[digit[0]]
    digit1 = _last_digit_2_str(digit[1]) if is_fina else _1_digit_2_str(digit[1])
    return (word2[digit[0]] + ' ' + digit1).strip()


def _3_digits_to_str(digit, is_fina=True):
    digstr = digit.lstrip('0')
    if len(digstr) == 0:
        return ''
    if len(digstr) == 1:
        return _1_digit_2_str(digstr)
    if len(digstr) == 2:
        return _2_digits_2_str(digstr, is_fina)
    if digit[-2:] == '00':
        return _1_digit_2_str(digit[0]) + ' зуу' if is_fina else _1_digit_2_str(digit[0]) + ' зуун'
    else:
        return _1_digit_2_str(digit[0]) + ' зуун ' + _2_digits_2_str(digit[-2:], is_fina)

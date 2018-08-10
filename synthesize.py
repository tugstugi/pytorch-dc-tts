#!/usr/bin/env python
"""Synthetize sentences into speech."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import sys
import argparse
from tqdm import *

import numpy as np
import torch

from models import Text2Mel, SSRN
from hparams import HParams as hp
from audio import save_to_wav
from utils import get_last_checkpoint_file_name, load_checkpoint, save_to_png

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", required=True, choices=['ljspeech', 'mbspeech'], help='dataset name')
args = parser.parse_args()

if args.dataset == 'ljspeech':
    from datasets.lj_speech import vocab, get_test_data

    SENTENCES = [
        "The birch canoe slid on the smooth planks.",
        "Glue the sheet to the dark blue background.",
        "It's easy to tell the depth of a well.",
        "These days a chicken leg is a rare dish.",
        "Rice is often served in round bowls.",
        "The juice of lemons makes fine punch.",
        "The box was thrown beside the parked truck.",
        "The hogs were fed chopped corn and garbage.",
        "Four hours of steady work faced us.",
        "Large size in stockings is hard to sell.",
        "The boy was there when the sun rose.",
        "A rod is used to catch pink salmon.",
        "The source of the huge river is the clear spring.",
        "Kick the ball straight and follow through.",
        "Help the woman get back to her feet.",
        "A pot of tea helps to pass the evening.",
        "Smoky fires lack flame and heat.",
        "The soft cushion broke the man's fall.",
        "The salt breeze came across from the sea.",
        "The girl at the booth sold fifty bonds."
    ]
else:
    from datasets.mb_speech import vocab, get_test_data

    SENTENCES = [
        "Нийслэлийн прокурорын газраас төрийн өндөр албан тушаалтнуудад холбогдох зарим эрүүгийн хэргүүдийг шүүхэд шилжүүлэв.",
        "Мөнх тэнгэрийн хүчин дор Монгол Улс цэцэглэн хөгжих болтугай.",
        "Унасан хүлгээ түрүү магнай, аман хүзүүнд уралдуулж, айрагдуулсан унаач хүүхдүүдэд бэлэг гардууллаа.",
        "Албан ёсоор хэлэхэд “Монгол Улсын хэрэг эрхлэх газрын гэгээнтэн” гэж нэрлээд байгаа зүйл огт байхгүй.",
        "Сайн чанарын бохирын хоолой зарна.",
        "Хараа тэглэх мэс заслын дараа хараа дахин муудах магадлал бага.",
        "Ер нь бол хараа тэглэх мэс заслыг гоо сайхны мэс засалтай адилхан гэж зүйрлэж болно.",
        "Хашлага даван, зүлэг гэмтээсэн жолоочийн эрхийг хоёр жилээр хасжээ.",
        "Монгол хүн бидний сэтгэлийг сорсон орон. Энэ бол миний төрсөн нутаг. Монголын сайхан орон.",
        "Постройка крейсера затягивалась из-за проектных неувязок, необходимости."
    ]

torch.set_grad_enabled(False)

text2mel = Text2Mel(vocab).eval()
last_checkpoint_file_name = get_last_checkpoint_file_name(os.path.join(hp.logdir, '%s-text2mel' % args.dataset))
# last_checkpoint_file_name = 'logdir/%s-text2mel/step-020K.pth' % args.dataset
if last_checkpoint_file_name:
    print("loading text2mel checkpoint '%s'..." % last_checkpoint_file_name)
    load_checkpoint(last_checkpoint_file_name, text2mel, None)
else:
    print("text2mel not exits")
    sys.exit(1)

ssrn = SSRN().eval()
last_checkpoint_file_name = get_last_checkpoint_file_name(os.path.join(hp.logdir, '%s-ssrn' % args.dataset))
# last_checkpoint_file_name = 'logdir/%s-ssrn/step-005K.pth' % args.dataset
if last_checkpoint_file_name:
    print("loading ssrn checkpoint '%s'..." % last_checkpoint_file_name)
    load_checkpoint(last_checkpoint_file_name, ssrn, None)
else:
    print("ssrn not exits")
    sys.exit(1)

# synthetize by one by one because there is a batch processing bug!
for i in range(len(SENTENCES)):
    sentences = [SENTENCES[i]]

    max_N = len(SENTENCES[i])
    L = torch.from_numpy(get_test_data(sentences, max_N))
    zeros = torch.from_numpy(np.zeros((1, hp.n_mels, 1), np.float32))
    Y = zeros
    A = None

    for t in tqdm(range(hp.max_T)):
        _, Y_t, A = text2mel(L, Y, monotonic_attention=True)
        Y = torch.cat((zeros, Y_t), -1)
        _, attention = torch.max(A[0, :, -1], 0)
        attention = attention.item()
        if L[0, attention] == vocab.index('E'):  # EOS
            break

    _, Z = ssrn(Y)

    Y = Y.cpu().detach().numpy()
    A = A.cpu().detach().numpy()
    Z = Z.cpu().detach().numpy()

    save_to_png('samples/%d-att.png' % (i + 1), A[0, :, :])
    save_to_png('samples/%d-mel.png' % (i + 1), Y[0, :, :])
    save_to_png('samples/%d-mag.png' % (i + 1), Z[0, :, :])
    save_to_wav(Z[0, :, :].T, 'samples/%d-wav.wav' % (i + 1))

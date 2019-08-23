"""Data loader for the a generic LJSpeech-style dataset. """
from .lj_speech import LJSpeech

class Generic(LJSpeech):
    def __init__(self, keys, dir_name):
        super().__init__(keys, dir_name=dir_name)

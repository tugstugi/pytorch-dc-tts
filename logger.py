"""Wrapper class for logging into the TensorBoard and comet.ml"""
__author__ = 'Erdene-Ochir Tuguldur'
__all__ = ['Logger']

import os
from tensorboardX import SummaryWriter

from hparams import HParams as hp


class Logger(object):

    def __init__(self, dataset_name, model_name):
        self.model_name = model_name
        self.project_name = "%s-%s" % (dataset_name, self.model_name)
        self.logdir = os.path.join(hp.logdir, self.project_name)
        self.writer = SummaryWriter(log_dir=self.logdir)

    def log_step(self, phase, step, loss_dict, image_dict):
        if phase == 'train':
            if step % 50 == 0:
                # self.writer.add_scalar('lr', get_lr(), step)
                # self.writer.add_scalar('%s-step/loss' % phase, loss, step)
                for key in sorted(loss_dict):
                    self.writer.add_scalar('%s-step/%s' % (phase, key), loss_dict[key], step)

            if step % 1000 == 0:
                for key in sorted(image_dict):
                    self.writer.add_image('%s/%s' % (self.model_name, key), image_dict[key], step)

    def log_epoch(self, phase, step, loss_dict):
        for key in sorted(loss_dict):
            self.writer.add_scalar('%s/%s' % (phase, key), loss_dict[key], step)

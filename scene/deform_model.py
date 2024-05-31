import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
# try using ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DeformModel:
    def __init__(self, is_blender=False, is_6dof=False):
        self.deform = DeformNetwork(is_blender=is_blender, is_6dof=is_6dof).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb):
        return self.deform(xyz, time_emb)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # record the initial learning rate
        self.init_lr = training_args.position_lr_init * self.spatial_lr_scale

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)
        
        # try using ReduceLROnPlateau
        self.scheduler_plateau = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=1e-5, min_lr=training_args.position_lr_final, verbose=True)
        # record min_lr
        self.min_lr = training_args.position_lr_final

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    # ADDED
    def load_zerodeform_weights(self, model_path):
        weights_path = os.path.join(model_path, "zero_deform.pth")
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # update learning rate using ReduceLROnPlateau
    def update_learning_rate_on_plateau(self, loss):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                self.scheduler_plateau.step(loss)
                lr = self.scheduler_plateau._last_lr[0]
                return lr
    
    def reset_scheduler(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                param_group['lr'] = self.init_lr
        self.scheduler_plateau = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=1e-5, min_lr=self.min_lr, verbose=True)

import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Union, Optional, Callable, Any
from torch.optim import Optimizer
from pytorch_lightning.core.optimizer import LightningOptimizer

import pytorch_lightning as pl
from lightning.pytorch.utilities import measure_flops

from NeuralNetwork.ModelTwoLayer import *
from NeuralNetwork.ModelResNet1D import *
from NeuralNetwork.ModelMLP import *
from NeuralNetwork.ModelLSTM import *
from NeuralNetwork.IMUNet import IMUNet
from NeuralNetwork.MobileNet import MobileNet
from NeuralNetwork.MobileNetV2 import MobileNetV2

from SmartPDR.DataMng.Dataset.FastDataset import *


class PDRModule(pl.LightningModule):
    def __init__(self, model_name="ResMLP",
                 model_para=None,
                 train_para=None,
                 data_para=None):
        super(PDRModule, self).__init__()

        self.train_para = train_para

        self.model_name = model_name
        self.model_para = model_para

        if "out_dim" in model_para:
            self.out_dim = model_para["out_dim"]

        self.data_para = data_para

        self.input_len = (
                data_para["data_len"]["before_len"]
                + data_para["data_len"]["after_len"]
                + data_para["data_len"]["len"]
        )
        self.example_input_array = torch.zeros([1, 6, self.input_len], device=self.device)

        self.switch_epoch = self.train_para["switch_epoch"]

        self.batch_size: int = train_para["batch_size"]

        self.net = self.set_network(self.model_name, self.model_para)

        self.lr = self.train_para['lr']

        self.measure_flops()

        self.save_hyperparameters()
        #
        # # 梯度裁剪：裁剪总范数不超过 0.1
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1, error_if_nonfinite=True)
        # optimizer.step(closure=optimizer_closure)

    def measure_flops(self):
        fwd_flops = measure_flops(self.net,
                                  lambda: self.net(self.example_input_array))
        # print(f"model flops: {fwd_flops / (1024 * 1024)} M Flops")
        return fwd_flops

    def set_network(self, model_name, model_para):
        if model_name == "ResNet":
            net =  ResNet1D(
                block_type=BasicBlock1D,
                in_dim=model_para["in_dim"],
                out_dim=model_para["out_dim"],
                group_sizes=model_para["group_sizes"],
                inter_dim=(model_para["pre_len"] + model_para["cur_len"] + model_para["after_len"]) // 32 + 1,
            )
            # return torch.compile(net)
            return net
        elif model_name == "MLP":
            return MLPCombineNet(model_para)
        elif model_name == "TwoLayerModel":
            return TwoLayerModel(model_para)
        elif model_name == "LSTM":
            return TLIOLSTM(in_dim=6, out_dim=6, hidden_size=model_para["hidden_units"],
                                layer_num=model_para["layer_num"])
        elif model_name == "IMUNet":
            return IMUNet()
        elif model_name == "MobileNet":
            return MobileNet()
        elif model_name == "MobileNetV2":
            return MobileNetV2()
        else:
            raise ValueError(f"unknown model name: {model_name}")

    def forward(self, x):
        y, y_cov = self.net.forward(x)
        y_all = torch.zeros([y.shape[0], (2*self.out_dim)])
        y_all[:, 0:self.out_dim] = y
        y_all[:, self.out_dim:(2*self.out_dim)] = y_cov
        return y_all

    def loss_distribution_diag(self, pred, pred_cov, traget):
        pred_cov = torch.maximum(pred_cov, np.log(1e-3) * torch.ones_like(pred_cov))
        loss = ((pred - traget).pow(2.0)) / (2.0 * torch.exp(2.0 * pred_cov)) + pred_cov
        return torch.mean(loss)

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Union[Optimizer, LightningOptimizer],
            optimizer_closure: Optional[Callable[[], Any]] = None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False
    ) -> None:
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=0.1, error_if_nonfinite=True
        )

        # 执行参数更新
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch:
            batch_idx:

        Returns:

        """
        # TODO: reload train dataset each epoch for offline data augmentation.
        # x, y = batch
        # y_pred, y_pred_cov = self.net(x)

        mse_loss, dis_loss, nll_loss = self.shared_step(batch)
        # nll_loss = self.loss_distribution_diag(y_pred, y_pred_cov, y)

        self.log("dis_loss", dis_loss, prog_bar=True)
        self.log("mse", mse_loss, prog_bar=True)
        self.log("nll_loss", nll_loss, prog_bar=True)

        if self.current_epoch < self.switch_epoch:
            return mse_loss  # + kl_loss
        else:
            return nll_loss  # + kl_loss

    def test_step(self, batch, batch_idx):
        mse_loss, dis_loss, nll_loss = self.shared_step(batch)
        self.log("test_dis_loss", dis_loss, prog_bar=True)
        self.log("test_mse", mse_loss, prog_bar=True)
        self.log("test_nll", nll_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        mse_loss, dis_loss, nll_loss = self.shared_step(batch)

        self.log("val_dis_loss", dis_loss, prog_bar=True)
        self.log("val_mse", mse_loss, prog_bar=True)
        self.log("val_nll", nll_loss, prog_bar=True)
        # return mse_loss

    def shared_step(self, batch):
        x, y = batch
        # print(f'x_shape: {x.shape}, y_shape: {y.shape}')
        y_pred, y_pred_cov = self.net(x)

        if self.current_epoch < self.switch_epoch:
            y_pred_cov = y_pred_cov.detach()

        mse_loss = F.mse_loss(y_pred, y[:, 0:self.out_dim])
        dis_loss = torch.sum((y_pred - y[:, 0:self.out_dim]).pow(2), 1).pow(0.5).mean()
        nll_loss = self.loss_distribution_diag(y_pred, y_pred_cov, y[:, 0:self.out_dim])
        return mse_loss, dis_loss, nll_loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(),
        #                              lr=self.lr, #self.train_para["lr"],
        #                              weight_decay=0.,
        #                              amsgrad=True)

        # optimizer = torch.optim.AdamW(self.parameters(),
        #                               lr=self.train_para["lr"],
        #                               amsgrad=True, weight_decay=0.)

        # optimizer = SAMSGD(self.parameters(), lr=self.train_para['lr'], rho=0.05)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.train_para["lr"])

        optimizer = torch.optim.Adam(self.parameters(), self.train_para["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12
        )

        return optimizer

    def train_dataloader(self):
        dataset = FastDataset('train', self.data_para, True)

        train_loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=torch.get_num_threads(),
            pin_memory=True,
            prefetch_factor=2
        )
        return train_loader

    def val_dataloader(self):
        print("load validation dataset")

        dataset = FastDataset('valid', self.data_para, False)
        return DataLoader(
            dataset, num_workers=64, shuffle=False, batch_size=self.batch_size
        )

    def test_dataloader(self):
        print("load test dataset")

        dataset = FastDataset('test', self.data_para, False)
        test_unknown_loader = DataLoader(
            dataset=dataset, num_workers=64, batch_size=self.batch_size, shuffle=False
        )
        return test_unknown_loader


if __name__ == '__main__':
    data_para = {
        'dataset_file_name': "IOSFullData_ForDL.h5",
        'data_len': {
            'before_len': 0,
            'len': 200,
            'after_len': 0
        },
        'step_len': 2,
        'aug': {
            'flag': True,
            'yaw_aug': True,
            'acc_bias_aug': {
                'flag': True,
                'val': 0.2,
            },
            'gyr_bias_aug': {
                'flag': True,
                'val': 0.5,  # deg
            },
            'acc_noise_aug': {
                'flag': True,
                'val': 0.05,
            },
            'gyr_noise_aug': {
                'flag': True,
                'val': 0.01,  # deg
            },
            'gravity_perturb': {
                'flag': True,
                'pos_perturb': False,
                'val': 10.,  # deg
            },
            'rigid_transformation': {
                'flag': False,
                't_val': [0.1, 0.1, 0.1]
            }
        }
    }

    model_name = "TwoLayerModel"
    model_para = {
        "input_len": (data_para['data_len']['before_len'] +
                      data_para['data_len']['len'] +
                      data_para['data_len']['after_len']),
        "input_channel": 6,
        "patch_len": 10,
        "feature_dim": 256,
        "out_dim": 3,
        "active_func": "GELU",
        "extractor": {
            "name": "ResMLP",
            "layer_num": 6,
            "expansion": 2,
            "dropout": 0.5  # probability of set some of the elements to zero.
        },
        "reg": {
            "name": "MeanMLP",
            "layer_num": 2,
        }
    }

    train_para = {"lr": 0.001, "batch_size": 256, 'switch_epoch': 10}

    pdr_model = PDRModule(model_name=model_name,
                          model_para=model_para,
                          data_para=data_para,
                          train_para=train_para)

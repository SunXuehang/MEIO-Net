import os
import sys
import time

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

import wandb

from SmartPDR.LitModel.PDRModule import *

from NeuralNetwork.EarlyExit.EarlyExitDense import *

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


class EarlyExistPDRModule(PDRModule):
    def __init__(
            self,
            model_name: str = "",
            model_para: Dict = None,
            train_para: Dict = None,
            data_para: Dict = None,
    ):
        """

        Args:
            model_name:
            model_para:
            train_para:
            data_para:
            learning_rate:
        """
        super(EarlyExistPDRModule, self).__init__(model_name=model_name,
                                                  model_para=model_para,
                                                  train_para=train_para,
                                                  data_para=data_para)

        ###########
        self.exit_port_num = model_para["layer_num"]
        self.output_dims = model_para["output_dims"]

        self.mse_power_n = train_para["mse_power_n"]
        self.nll_power_n = train_para["nll_cov_power_n"]
        if 'warm_up_epoch' in train_para:
            self.warm_up_epoch = train_para["warm_up_epoch"] # 20
        else:
            self.warm_up_epoch = 0
        if 'same_weight' in train_para:
            self.same_weight = train_para["same_weight"]
        else:
            self.same_weight = False

        if 'useful_layer_num' in model_para:
            self.useful_layer_num = model_para["useful_layer_num"]
        else:
            self.useful_layer_num = self.exit_port_num
        if self.useful_layer_num > self.exit_port_num:
            raise Exception("useful_layer_num should be less than exit_port_num")
        self.start_useful_layer = self.exit_port_num - self.useful_layer_num

        self.mse_weight_array = self.gen_scale_array(
            self.exit_port_num, self.mse_power_n
        )
        self.nll_cov_weight_array = self.gen_scale_array(
            self.exit_port_num, self.nll_power_n
        )

        # 用于保存每个batch中，位移误差最小的exit_port，模型收敛后，损失函数值计算之前的出口
        self.loss_switch_epoch = None
        self.min_disp_index = None
        self.loss_switch_epoch = train_para.get('switch_loss_epoch', self.loss_switch_epoch)

        # print(f" mse weight list: {self.mse_weight_array}")
        # print(f" nll cov weight list: {self.nll_cov_weight_array}")
        ###############

        ###############

    def measure_flops(self):
        fwd_flops = measure_flops(self.net, lambda: self.net.forward_with_exit(self.example_input_array))
        # print(f"model flops: {fwd_flops / (1024 * 1024)} M Flops")
        return fwd_flops

    def measure_chosen_exit_flops(self, chosen_exit_index):
        fwd_flops = measure_flops(self.net, lambda: self.net.forward_with_choosen_exit(self.example_input_array, chosen_exit_index))
        # print(f"model exit{chosen_exit_index} flops: {fwd_flops / (1024 * 1024)} M Flops")
        return fwd_flops

    def gen_scale_array(self, layer_num: int, n_pow: float):
        if not self.same_weight:
            common_weight = [float(i) / float(layer_num) for i in range(1, layer_num + 1)]
            weight = torch.tensor([(common_weight[i] ** n_pow) for i in range(layer_num)])
        else:
            weight = torch.tensor([1.0 for i in range(layer_num)])

        # 归一化使权重和为1
        weight_sum = torch.sum(weight)
        if weight_sum > 0:  # 防止除以零
            weight = weight / weight_sum

        return weight

    def set_network(self, model_name, model_para):
        # print(f"set network {model_name}")
        if model_name == "EarlyExitDense":
            return InplaceEarlyExitDense(model_para)
        else:
            raise Exception(f"Unknown network name {model_name}")

    def forward(self, x):
        y = self.net.forward_with_exit(x)
        y_all = torch.zeros([y[0].shape[0], self.output_dims[0]*2])
        y_all[:, 0:self.output_dims[0]] = y[0][:, :]
        y_all[:, self.output_dims[0]:6] = y[1][:,:]
        return y_all

    def forward_to_end(self, x):
        y = self.net.forward_to_end(x)
        y_all = torch.zeros([y[0].shape[0], self.output_dims[0]*2])
        y_all[:, 0:self.output_dims[0]] = y[0][:, :]
        y_all[:, self.output_dims[0]:6] = y[1][:,:]
        return y_all

    def forward_with_choosen_exit(self, x, exit_index):
        y = self.net.forward_with_choosen_exit(x, exit_index)
        y_all = torch.zeros([y[0][0].shape[0], self.output_dims[0]*2])
        y_all[:, 0:self.output_dims[0]] = y[0][0][:, :]
        y_all[:, self.output_dims[0]:6] = y[0][1][:,:]
        return y_all

    def forward_to_get_feature(self, x, exit_index):
        feature = self.net.forward_to_get_feature(x, exit_index)
        return feature

    def full_forward(self, x):
        head_out_list, y_list = self.net.full_forward(x)
        processed_output_list = []
        for i in range(len(y_list)):
            y = y_list[i]
            y_all = torch.zeros([y[0][0].shape[0], self.output_dims[0] * 2])
            y_all[:, 0:self.output_dims[0]] = y[0][0][:, :]
            y_all[:, self.output_dims[0]:6] = y[0][1][:, :]
            # 将处理后的结果添加到 list 中
            processed_output_list.append(y_all)

        # 返回所有处理后的输出
        return processed_output_list

    def weighted_loss_distribution_diag(self, pred, pred_cov, target, cov_weight):
        mah_loss = ((pred - target).pow(2.0)) / (2.0 * torch.exp(2.0 * pred_cov))
        return torch.mean(mah_loss + pred_cov), torch.mean(
            mah_loss + cov_weight * pred_cov
        )
        # return torch.mean(mah_loss + pred_cov), torch.mean(
        #     mah_loss + 0.1 * cov_weight * pred_cov
        # )

    def shared_step(self, x, y, full_record_flag= False):

        mse_loss_array = torch.zeros(self.exit_port_num)
        nll_loss_array = torch.zeros(self.exit_port_num)
        weighted_nll_loss_array = torch.zeros(self.exit_port_num)
        dis_loss_array = torch.zeros(self.exit_port_num)

        tensor_logger = None
        wandb_logger = None
        if full_record_flag:
            tensor_logger = self.get_tensorboard_logger()
            wandb_logger = self.get_wandb_logger()

        _, real_output_list = self.net.full_forward(x)

        # 获取 batch 大小
        batch_size = real_output_list[0][0][0].shape[0]
        # 初始化最小误差索引
        min_disp_index = torch.zeros(batch_size, dtype=torch.long)
        # 计算每个 batch 的误差最小的 exit_port
        for batch in range(batch_size):
            target_disp = y[batch, :self.output_dims[0]].detach()
            all_disp = torch.stack([real_output_list[layer_idx][0][0][batch].detach() for layer_idx in range(self.exit_port_num)])
            min_disp_index[batch] = torch.argmin(F.mse_loss(all_disp, target_disp.unsqueeze(0).expand_as(all_disp), reduction='none').mean(dim=1))

        for layer_idx in range(self.exit_port_num):
            y_pred, y_pred_cov = real_output_list[layer_idx][0]

            if self.current_epoch < self.switch_epoch:
                y_pred_cov = y_pred_cov.detach()

            # if self.loss_switch_epoch:
            #     if self.current_epoch >= self.loss_switch_epoch:
            #         mse_loss = F.mse_loss(y_pred, y[:, :self.output_dims[0]], reduction='none').mean(dim=1)
            #         valid_mse = [mse_loss[b] for b in range(batch_size) if layer_idx <= min_disp_index[b]]
            #         if valid_mse:
            #             mse_loss_array[layer_idx] = torch.stack(valid_mse).mean()
            #         else:
            #             mse_loss_array[layer_idx] = torch.tensor(0.0)
            #     else:
            #         mse_loss_array[layer_idx] = F.mse_loss(y_pred, y[:, 0:self.output_dims[0]])
            # else:
            #     mse_loss_array[layer_idx] = F.mse_loss(y_pred, y[:, 0:self.output_dims[0]])

            mse_loss_array[layer_idx] = F.mse_loss(y_pred, y[:, 0:self.output_dims[0]])
            dis_loss_raw = torch.sum((y_pred - y[:, 0:self.output_dims[0]]).pow(2), 1).pow(0.5).detach()
            dis_loss_array[layer_idx] = (
                dis_loss_raw.mean()
            )
            # if not tensor_logger is None:
            #     tensor_logger.add_histogram(tag=f"dis_loss_{layer_idx}",
            #                                 values = dis_loss_raw,
            #                                 global_step=self.global_step)
            if (not wandb_logger is None) and (self.global_step % 1000 == 0 and self.global_step > 10000):
                wandb.log({f"dis_loss_{layer_idx}": wandb.Histogram(dis_loss_raw.cpu())})

            nll_loss_array[layer_idx], weighted_nll_loss_array[layer_idx] = (
                self.weighted_loss_distribution_diag(
                    y_pred, y_pred_cov, y[:, 0:self.output_dims[0]], self.nll_cov_weight_array[layer_idx]
                )
            )

        return mse_loss_array, dis_loss_array, nll_loss_array, weighted_nll_loss_array

    def get_tensorboard_logger(self):
        # for logger in self.loggers:
        #     if isinstance(logger, TensorBoardLogger):
        #         return logger.experiment
        return None

    def get_wandb_logger(self):
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                return logger.experiment
        return None

    def training_step(self, batch, batch_idx):
        x, y = batch
        mse_loss_list, dis_loss_list, nll_loss_list, weighted_nll_loss_array = (
            self.shared_step(x = x, y = y)
        )

        # if self.global_step % 1000 == 0:
        #     tensor_logger = self.get_tensorboard_logger()
        #     if not tensor_logger is None:
        #         tensor_logger.add_histogram("dis_loss_array", dis_loss_list, self.global_step)
        #         tensor_logger.add_histogram("mse_loss_array", mse_loss_list, self.global_step)
        #         tensor_logger.add_histogram("nll_loss_array", nll_loss_list, self.global_step)
        #         dis_loss_dictionary = {str(k): v for k, v in enumerate(dis_loss_list)}
        #         tensor_logger.add_scalars(main_tag="multi_dis_loss",
        #                                   tag_scalar_dict=dis_loss_dictionary,
        #                                   global_step=self.global_step)

        #
        # self.log("dis_loss", torch.mean(dis_loss_list[self.start_useful_layer:]))
        # self.log("mse_loss", torch.mean(mse_loss_list[self.start_useful_layer:]))
        # self.log("nll_loss", torch.mean(nll_loss_list[self.start_useful_layer:]))

        for i in range(self.exit_port_num):
            self.log(f"dis_loss_{i}", dis_loss_list[i])
            self.log(f"mse_loss_{i}", mse_loss_list[i])
            self.log(f"nll_loss_{i}", nll_loss_list[i])

        alpha = 1.0 - (self.current_epoch - self.switch_epoch) / self.warm_up_epoch
        # 确保 alpha 在 [0, 1] 范围内
        alpha = max(0.0, min(1.0, alpha))

        if self.current_epoch < self.switch_epoch:
            return_loss = torch.mean(mse_loss_list[self.start_useful_layer:] * self.mse_weight_array[self.start_useful_layer:])
        else:
            return_loss = (alpha * torch.mean(mse_loss_list[self.start_useful_layer:] * self.mse_weight_array[self.start_useful_layer:]) +
                    (1.0 - alpha) * torch.mean(weighted_nll_loss_array[self.start_useful_layer:]))

        self.log("return_loss", return_loss)

        return return_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mse_loss_list, dis_loss_list, nll_loss_list, weighted_nll_loss_list = (
            self.shared_step(x, y)
        )

        # tensor_logger = self.get_tensorboard_logger()
        # if not tensor_logger is None:
        #     tensor_logger.add_scalars(main_tag="valid_multi_dis_loss",
        #                               tag_scalar_dict={str(k):v for k, v in enumerate(dis_loss_list)},
        #                               global_step=self.global_step)

        # self.log("val_dis_loss", torch.mean(dis_loss_list[self.start_useful_layer:]), prog_bar=True)
        self.log("val_dis_loss", dis_loss_list[-1])
        # self.log("val_mse_loss", torch.mean(mse_loss_list[self.start_useful_layer:]))
        # self.log("val_nll_loss", torch.mean(nll_loss_list[self.start_useful_layer:]))

        for i in range(self.exit_port_num):
            self.log(f"val_dis_loss_{i}", dis_loss_list[i])
            self.log(f"val_mse_loss_{i}", mse_loss_list[i])
            self.log(f"val_nll_loss_{i}", nll_loss_list[i])

    def test_step(self, batch, batch_idx):
        x, y = batch
        mse_loss_list, dis_loss_list, nll_loss_list, weighted_nll_loss_list = (
            self.shared_step(x, y)
        )

        # self.log("test_dis_loss", torch.mean(dis_loss_list[self.start_useful_layer:]), prog_bar=True)
        self.log("test_dis_loss", dis_loss_list[-1], prog_bar=True)
        # self.log("test_mse_loss", torch.mean(mse_loss_list[self.start_useful_layer:]))
        # self.log("test_nll_loss", torch.mean(nll_loss_list[self.start_useful_layer:]))

        for i in range(self.exit_port_num):
            self.log(f"test_dis_loss_{i}", dis_loss_list[i], prog_bar=True)
            self.log(f"test_mse_loss_{i}", mse_loss_list[i])
            self.log(f"test_nll_loss_{i}", nll_loss_list[i])


if __name__ == "__main__":
    data_para = {
        "dataset_file_name": "IOSFullData_ForDL.h5",
        "data_len": {"before_len": 0, "len": 200, "after_len": 0},
        "step_len": 2,
        "aug": {
            "flag": True,
            "yaw_aug": True,
            "acc_bias_aug": {
                "flag": True,
                "val": 0.2,
            },
            "gyr_bias_aug": {
                "flag": True,
                "val": 0.5,  # deg
            },
            "acc_noise_aug": {
                "flag": True,
                "val": 0.05,
            },
            "gyr_noise_aug": {
                "flag": True,
                "val": 0.01,  # deg
            },
            "gravity_perturb": {
                "flag": True,
                "pos_perturb": False,
                "val": 10.0,  # deg
            },
            "rigid_transformation": {"flag": False, "t_val": [0.1, 0.1, 0.1]},
        },
    }

    model_name = "EarlyExitDense"
    model_para = {
        "input_len": (
                data_para["data_len"]["before_len"]
                + data_para["data_len"]["len"]
                + data_para["data_len"]["after_len"]
        ),
        "input_channel": 6,
        "patch_len": 10,
        "feature_dim": 512,
        "output_dims": [3, 1],
        "active_function": "GELU",
        "layer_num": 6,
        "noise_variance": 1e-3,
        "backbone": {
            "name": "ResMLP",
            "expansion_factor": 2,
            "dropout_rate": 0.5,  # probability of set some of the elements to zero.
        },
        "reg": {
            "name": "MeanMLP",
            "layer_num": 2,
        },
    }

    train_para = {
        "lr": 0.001,
        "batch_size": 256,
        "switch_epoch": 10,
        "mse_power_n": 1,
        "nll_cov_power_n": 0.5,
    }

    pdr_model = EarlyExistPDRModule(
        model_name=model_name,
        model_para=model_para,
        data_para=data_para,
        train_para=train_para,
    )
    # pdr_model = pdr_model.to("cuda")

    # pdr_model.shared_step(
    #     torch.zeros([128, 6, 100], device="cuda"), torch.zeros([128, 3], device="cuda")
    # )
    # pdr_model.training_step(
    #     (
    #         torch.zeros([128, 6, 100], device="cuda"),
    #         torch.zeros([128, 3], device="cuda"),
    #     ),
    #     1,
    # )

    y = pdr_model.forward_with_choosen_exit(torch.zeros([128, 6, 200]), 1)

    pdr_model.measure_flops()
    # pdr_model.measure_chosen_exit_flops(1)

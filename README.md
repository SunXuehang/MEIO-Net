# MEIO-Net
Code for paper: "MEIO-Net: A Motion-Aware Early-Exit Inertial Odometry Network for Efficient Pedestrian Dead Reckoning (PDR)"

## Prerequisites

Install dependency use pip:
```bash
pip install torch einops numpy
```

## Usage
The LLIO contained in the model_twolayer.py, and a unit test for illustration input and output are provided in this file.
```python
if __name__ == "__main__":
    data_para = {
        "dataset_file_name": "IOSFullData_ForDL.h5",
        "data_len": {"before_len": 0, "len": 100, "after_len": 0},
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
        "feature_dim": 128,
        "output_dims": [3, 1],
        "active_function": "GELU",
        "layer_num": 6,
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
    pdr_model = pdr_model.to("cuda")

    pdr_model.shared_step(
        torch.zeros([128, 6, 100], device="cuda"), torch.zeros([128, 3], device="cuda")
    )
    pdr_model.training_step(
        (
            torch.zeros([128, 6, 100], device="cuda"),
            torch.zeros([128, 3], device="cuda"),
        ),
        1,
    )

    y = pdr_model.forward([128, 6, 100], device='cuda')
```

## Acknowledgements
Thanks for LLIO [https://github.com/i2Nav-WHU/LightweightLearnedInertialOdometer.git].

## License
The source code is released under GPLv3 license.

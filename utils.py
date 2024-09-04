from monai.transforms import AsDiscrete, Compose, Transform
from monai.visualize.utils import matshow3d, blend_images
from monai.data import decollate_batch, NibabelWriter
from monai.inferers import sliding_window_inference
from scipy.spatial.transform import Rotation
from dacite import Config as DaciteConfig
from monai.utils import set_determinism
import monai.transforms as transforms
import matplotlib.patches as patches
from monai.metrics import DiceMetric
from torch.nn import functional as F
from matplotlib.patches import Patch
from monai.losses import DiceLoss
import matplotlib.pyplot as plt
from torchmetrics import Dice
from dacite import from_dict
import torch.optim as optim
from copy import deepcopy
from pathlib import Path
import lightning as l
import torch.nn as nn
import torchio as tio
from tqdm import tqdm
import torchmetrics
import numpy as np
import random
import shutil
import torch
import yaml
import cv2


class Config:

    batch_size = 1
    overlap = 0.50
    in_channels = 1
    out_channels = 14
    img_size = (96, 96, 96)
    feature_size = 16
    hidden_size = 768
    mlp_dim = 3072
    num_heads = 12
    pos_embed = 'perceptron'
    norm_name = 'instance'
    conv_block = True
    res_block = True
    dropout_rate = 0.0

    def __init__(self, config_path):
        
        config = parse_config(config_path)
        for key, value in config.items():
            setattr(self, key, value)

        self.l_device = [int(self.device[-1])]

    def init(self):

        set_determinism(self.random_state)
        seed_everything(self.random_state)
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('high')


def type_to_parser(type):

    if type == "Sequential":
        return load_sequential
    if type == "xLSTM":
        return load_xlstm
    return load_torch_module


def load_sequential(config):

    layers = []
    for layer_config in config["layers"]:
        layer = type_to_parser(layer_config["type"])(layer_config)
        layers.append(layer)
    
    return nn.Sequential(*layers)


def load_torch_module(config):

    layer_type = None
    for module in [nn]:
        try:
            layer_type = getattr(module, config["type"])
        except:
            continue

    if layer_type is None:
        raise ValueError(f"Wrong model type: {config['type']}")
    
    parameters = {key: value for key, value in list(config.items())[1:]}
    layer = layer_type(**parameters)
    
    return layer


def load_xlstm(config):

    pass
    # cfg = from_dict(data_class=xLSTMLMModelConfig, data=config["params"], config=DaciteConfig(strict=True))
    # return xLSTMLMModel(cfg)


def build_model_from_config(
        config
):

    return type_to_parser(config["type"])(config)


def parse_config(
        config_path
):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def seed_everything(seed):

    torch.backends.cudnn.deterministic = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def convert_angles(config):

    config.rotation_angles = [None for _ in range(3)]
    for i in range(3):
        config.rotation_angles[i] = config.rotation_degrees[i] * np.pi / 180


def save_comparison(config, segmentation, ground_truth):

    path_to_save = config.path_to_output.joinpath("slice_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].set_title("Segmentation slice")
    axes[1].set_title("Ground truth slice")
    axes[0].imshow(blend_images(**segmentation)[:, :, 170, :].permute(2, 1, 0).numpy())
    axes[1].imshow(blend_images(**ground_truth)[:, :, 170, :].permute(2, 1, 0).numpy())
    
    plt.savefig(path_to_save, bbox_inches='tight')


def save_slices_fig(config, data, frame_dim, title):

    fig = plt.figure()
    path_to_save = config.path_to_output.joinpath(title.lower().replace(" ", "_") + ".png")

    matshow3d(blend_images(**data), channel_dim=0, frame_dim=frame_dim, every_n=9, margin=5, title=title, figsize=config.figsize);
    plt.savefig(path_to_save, bbox_inches='tight')


def cut_volume(data):

    data_ = deepcopy(data)
    data_ = data_[:, 100 : -101, 100 : -101, 100 : -101]
    # data_["image"] = data_["image"][..., 100 : -101, 100 : -101, 100 : -101]
    # data_["label"] = data_["label"][..., 100 : -101, 100 : -101, 100 : -101]

    return data_


def get_start_slice_z(n_slices, v1, v2, v3):

    for slc in range(n_slices):
        if torch.all(v1[..., slc * 10] == 0) and torch.all(v2[..., slc * 10] == 0) and torch.all(v3[..., slc * 10] == 0):
            continue
        return slc
    

def get_start_slice_y(n_slices, v1, v2, v3):

    for slc in range(n_slices):
        if torch.all(v1[..., slc * 10, :] == 0) and torch.all(v2[..., slc * 10, :] == 0) and torch.all(v3[..., slc * 10, :] == 0):
            continue
        return slc


def save_slices(config, volume, segmentation, inversed_gt, gt):

    path_to_save = config.path_to_output.joinpath("slices_comparison.png")
    box_size = volume.size(-1)
    n_slices = (box_size - 201) // 10
    blend_1col = cut_volume(blend_images(volume, segmentation))
    blend_2col = cut_volume(blend_images(volume, inversed_gt))
    blend_3col = cut_volume(blend_images(volume, gt))

    start_slc = get_start_slice_z(n_slices, blend_1col, blend_2col, blend_3col)
    fig, axes = plt.subplots(n_slices - start_slc, 3, figsize=config.figsize)
    axes[0][0].set_title("Сегментация на повернутом объеме")
    axes[0][1].set_title("Сегментация на объеме")
    axes[0][2].set_title("Истинно верная карта классов")

    for slc in range(start_slc, n_slices):
        
        axes[slc - start_slc][0].imshow(blend_1col[..., slc * 10].permute(2, 1, 0).numpy())
        axes[slc - start_slc][1].imshow(blend_2col[..., slc * 10].permute(2, 1, 0).numpy())
        axes[slc - start_slc][2].imshow(blend_3col[..., slc * 10].permute(2, 1, 0).numpy())
        axes[slc - start_slc][0].set_ylabel(f"{slc * 10} срез по третей оси")

    plt.savefig(path_to_save, bbox_inches='tight')
    path_to_save = config.path_to_output.joinpath("slices_comparison_another_axes.png")
    start_slc = get_start_slice_y(n_slices, blend_1col, blend_2col, blend_3col)
    fig, axes = plt.subplots(n_slices - start_slc, 3, figsize=config.figsize)
    axes[0][0].set_title("Сегментация на повернутом объеме")
    axes[0][1].set_title("Сегментация на объеме")
    axes[0][2].set_title("  ")

    for slc in range(start_slc, n_slices):
        
        axes[slc - start_slc][0].imshow(blend_1col[..., slc * 10, :].permute(2, 1, 0).numpy())
        axes[slc - start_slc][1].imshow(blend_2col[..., slc * 10, :].permute(2, 1, 0).numpy())
        axes[slc - start_slc][2].imshow(blend_3col[..., slc * 10, :].permute(2, 1, 0).numpy())
        axes[slc - start_slc][0].set_ylabel(f"{slc * 10} срез по второй оси")

    plt.savefig(path_to_save, bbox_inches='tight')


def save_segmentation_samples(config, model, volume, rotated_volume, ground_truth, inversed_ground_truth, inverse_crop):

    model.eval()
    model.to(config.device)

    post_pred = transforms.Compose(
        [
            transforms.EnsureType("tensor", device="cpu"),
            transforms.AsDiscrete(argmax=True)
        ]
    )

    with torch.no_grad():

        rotated_output = sliding_window_inference(rotated_volume.unsqueeze(0).to(config.device), config.img_size, 4, model, overlap=config.overlap)
        rotated_output = post_pred(rotated_output[0])
        inversed = inverse_crop(
            {
                "image": torch.tensor([[[[0.]]]]),
                "label": rotated_output
            }
        )

        output = sliding_window_inference(volume.unsqueeze(0).to(config.device), config.img_size, 4, model, overlap=config.overlap)
        output = post_pred(output[0])

        save_slices(config, volume, inversed["label"], output, ground_truth)


class CutBorder(Transform):

    def __init__(self, keys, border_size):
        # super(CutBorder, self).__init__()

        self.keys = keys
        self.border_size = border_size

    def __call__(self, data):

        data = deepcopy(data)
        for key in self.keys:
            data[key] = data[key][
                ...,
                self.border_size : -self.border_size,
                self.border_size : -self.border_size,
                self.border_size : -self.border_size
            ]
        return data


class RotateSegmentationd(Transform):

    def __init__(self, keys, num_classes, angle, mode, inverse=False):
        # super(RotateSegmentationd, self).__init__()

        self.keys = keys
        self.mode = mode
        self.num_classes = num_classes
        self.post_pred = transforms.AsDiscrete(argmax=True)
        self.post_label = transforms.AsDiscrete(to_onehot=14)
        self.rotation = RotateVolume(angle, inverse=inverse, mode=mode)

    def __call__(self, data):

        x = data[self.keys]

        if self.mode == "bilinear":

            x_rotated = []
            x_rotated = self.post_label(x)
            x_rotated = self.rotation(x_rotated)
            x_rotated = self.post_pred(x_rotated).to(torch.uint8)
            # for cl in range(0, self.num_classes):
            #     class_segmentation = (x == cl).to(torch.uint8)
            #     x_rotated.append(
            #         self.rotation(class_segmentation)
            #     )

            # x_rotated = torch.cat(x_rotated)
            # x_rotated = self.post_pred(x_rotated).to(torch.uint8)
            assert x.shape == x_rotated.shape

        elif self.mode == "nearest":

            x_rotated = self.rotation(x)

        else:
            raise ValueError("Wrong interpolation mode")
        
        data[self.keys] = x_rotated
        return data
    

class RotateVolumed(Transform):

    def __init__(self, keys, angle, mode, inverse=False):
        # super(RotateVolumed, self).__init__()

        self.keys = keys
        self.rotation = RotateVolume(angle, mode, inverse=inverse)

    def __call__(self, data):

        data[self.keys] = self.rotation(data[self.keys])
        return data
    
    def inverse(self, data):

        data[self.keys] = self.rotation.inverse(data[self.keys])
        return data
    

class RotateVolume(Transform):

    def __init__(self, angle, mode, inverse=False):
        # super(RotateVolume, self).__init__()

        self.rotation = transforms.Compose(
            [
                transforms.Rotate((angle[0], 0, 0), mode=mode, keep_size=True) if angle[0] != 0 else transforms.Identity(),
                transforms.Rotate((0, angle[1], 0), mode=mode, keep_size=True) if angle[1] != 0 else transforms.Identity(),
                transforms.Rotate((0, 0, angle[2]), mode=mode, keep_size=True) if angle[2] != 0 else transforms.Identity()
            ]
        ) if not inverse else transforms.Compose(
            [
                transforms.Rotate((0, 0, angle[2]), mode=mode, keep_size=True) if angle[2] != 0 else transforms.Identity(),
                transforms.Rotate((0, angle[1], 0), mode=mode, keep_size=True) if angle[1] != 0 else transforms.Identity(),
                transforms.Rotate((angle[0], 0, 0), mode=mode, keep_size=True) if angle[0] != 0 else transforms.Identity()
            ]
        )

    def __call__(self, data):

        return self.rotation(data)
    
    def inverse(self, data):

        return self.rotation.inverse(data)


def calc_mse(x_dataset, y_dataset):
    mse_losses = []

    for i in tqdm(range(len(x_dataset))):
        x, y = x_dataset[i]["image"], y_dataset[i]["image"]

        mse_losses.append(F.mse_loss(x, y).item())

    return np.mean(mse_losses)


def calc_scale_coef(volume1, volume2):

    non_zero = (volume1 != 0) + (volume2 != 0)
    non_zero = (non_zero != 0).sum().item()

    return volume1.shape.numel() / non_zero


def calc_mae(x_dataset, y_dataset):
    mae = torchmetrics.MeanAbsoluteError()
    mae_losses = []

    for i in tqdm(range(len(x_dataset))):
        x, y = x_dataset[i]["image"], y_dataset[i]["image"]

        mae_losses.append(mae(x, y).item() * calc_scale_coef(x, y))

    return np.mean(mae_losses)


def calc_dice_labels(x_dataset, y_dataset):

    post_label = Compose(
        [
            transforms.EnsureType("tensor", device="cpu"),
            transforms.AsDiscrete(to_onehot=14)
        ]
    )
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_metrics = []

    for i in tqdm(range(len(x_dataset))):
        x, y = x_dataset[i]["label"], y_dataset[i]["label"]
        x = post_label(x)
        y = post_label(y)
        dice_metric(y_pred=y, y=x)
        dice_metrics.append(dice_metric.aggregate().item())
        dice_metric.reset()

    return np.mean(dice_metrics)    


def calc_dice_segmentation(config, model, dataset):

    model.eval()
    model.to(config.device)

    post_pred = Compose(
        [
            transforms.EnsureType("tensor", device="cpu"),
            transforms.AsDiscrete(argmax=True, to_onehot=14)
        ]
    )
    post_label = Compose(
        [
            transforms.EnsureType("tensor", device="cpu"),
            transforms.AsDiscrete(to_onehot=14)
        ]
    )
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_metrics = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            batch = dataset[i]

            x, y = batch["image"].to(config.device), batch["label"]
            o = sliding_window_inference(x.unsqueeze(0), config.img_size, 4, model, overlap=config.overlap)

            outputs = post_pred(o[0])
            labels = post_label(y)
            dice_metric(y_pred=outputs, y=labels)
            dice_metrics.append(dice_metric.aggregate().item())
            dice_metric.reset()

    return np.mean(dice_metrics)


class MedNeXtModule(l.LightningModule):

    def __init__(self, model, patch_size, optimizer_config, predict_classes, overlap, ckpt_path=None, full_dataset=None):
        super().__init__()

        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.sw_batch_size = 1
        self.optimizer_config = optimizer_config
        self.ckpt_path = ckpt_path
        self.full_dataset = full_dataset
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose(
            [
                # EnsureType("tensor", device="cpu"),
                AsDiscrete(argmax=True, to_onehot=predict_classes)
            ]
        )
        self.post_label = Compose(
            [
                # EnsureType("tensor", device="cpu"),
                AsDiscrete(to_onehot=predict_classes)
            ]
        )
        self.ckpth_metadata = []
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.validation_step_outputs = []
        self.train_losses = []
        self.eval_losses = []

    def forward(self, x):
        return sliding_window_inference(x, self.patch_size, self.sw_batch_size, self.model.forward, overlap=self.overlap, sw_device=self.device)

    def training_step(self, batch):

        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        self.train_losses.append(loss.item())

        return loss

    def validation_step(self, batch):

        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)

        return d

    def on_validation_epoch_end(self):

        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].item()
            num_items += output["val_number"]
        mean_eval_dice = self.dice_metric.aggregate().item()
        mean_eval_loss = torch.tensor(val_loss / num_items)
        mean_train_loss = np.mean(self.train_losses)
        
        self.log("Eval Dice", mean_eval_dice)
        self.log("Eval Loss",mean_eval_loss)
        self.log("Train Loss", mean_train_loss)
        self.eval_loss = mean_eval_loss
        self.eval_dice = mean_eval_dice
        
        self.dice_metric.reset()
        self.train_losses.clear()
        self.validation_step_outputs.clear()

    def generate_path(self):

        path = Path(self.ckpt_path)
        # print(path, end=", ")
        path = path.joinpath(f"epoch={self.current_epoch}_dice={self.eval_dice:.3f}")
        # print(path)

        return path

    def delete_worst_epoch(self):

        self.ckpth_metadata = sorted(self.ckpth_metadata, key=lambda x: x["dice"])
        worst_epoch = self.ckpth_metadata.pop(0)
        # print(f"Removed dir - {worst_epoch['path']}")
        # worst_epoch["path"].rmdir()
        shutil.rmtree(worst_epoch["path"])

    def save_epoch(self, path):

        # print(f"Made dir - {path}")
        path.mkdir(parents=True, exist_ok=True)
        self.ckpth_metadata.append(
            deepcopy({
                "path": path,
                "epoch": self.current_epoch,
                "dice": self.eval_dice
            })
        )

    def on_save_checkpoint(self, checkpoint):

        if self.ckpt_path is not None:
            
            with torch.no_grad():
                path = self.generate_path()
                if len(self.ckpth_metadata) == 3:
                    self.delete_worst_epoch()
                self.save_epoch(path)
                # print(f"Folders - {len(self.ckpth_metadata)}")

                for i in range(len(self.full_dataset)):
                    d = self.full_dataset[i]
                    x, n = d["image"], d["image_name"]
                    segm_path = path.joinpath(n)

                    x = x.unsqueeze(0)
                    o = self.forward(x).squeeze(0)
                    writer = NibabelWriter()
                    writer.set_data_array(o, channel_dim=0)
                    # print(f"Writing - {segm_path}")
                    writer.write(segm_path, verbose=False)
                    o, x = o.cpu(), x.cpu()
    
    def configure_optimizers(self):
        
        opt_type = getattr(optim, self.optimizer_config["name"])
        opt = opt_type(self.model.parameters(), **self.optimizer_config["parameters"])

        return {
            "optimizer": opt,
            # "lr_scheduler": {
            #     "scheduler": sch,
            #     "interval": "step",
            #     "frequency": 1
            # }
        }
    

class CustomDataset():

    def __init__(self, metadata, angle_range, n_rotations, add_padding, border):

        self.metadata = metadata
        self.base = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")
            ),
            # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.BorderPadd(
                keys=["image"], spatial_border=border // 2, value=-175
            ) if add_padding else transforms.Identity(),
            transforms.BorderPadd(
                keys=["label"], spatial_border=border // 2, value=0
            ) if add_padding else transforms.Identity()
        ])
        self.preprocessing = transforms.Compose([
            transforms.ScaleIntensityRanged(
                keys="image", a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.EnsureTyped(keys="label", data_type="tensor", device="cpu"),
            transforms.AsDiscreted(keys="label", to_onehot=14)
        ])

        self.n_rotations = n_rotations
        self.angle_range = angle_range
        
        # Generate n rotation vectors with norm equal to degrees from angle_range
        directions = np.random.uniform(low=-1.0, high=1.0, size=(n_rotations, 3))
        directions /= np.linalg.norm(directions, axis=-1)[:, np.newaxis]
        angles = np.random.uniform(low=angle_range[0], high=angle_range[1], size=n_rotations)
        self.rotation_vectors = angles[:, np.newaxis] * directions

        # Generate rotation matrices from rotation vectors
        self.rotation_matrices = Rotation.from_rotvec(self.rotation_vectors, degrees=True).as_matrix()
        inverse_rotation_matrices = Rotation.from_rotvec(-self.rotation_vectors, degrees=True).as_matrix()
        
        # Add padding to rotation matrices
        self.rotation_matrices = np.pad(
            self.rotation_matrices,
            ((0, 0), (0, 1), (0, 1)),
            mode='constant',
            constant_values=0
        )
        inverse_rotation_matrices = np.pad(
            inverse_rotation_matrices,
            ((0, 0), (0, 1), (0, 1)),
            mode='constant',
            constant_values=0
        )

        self.rotations = [
            transforms.Affined(
                keys=["image", "label"],
                mode=["bilinear", "nearest"],
                affine=self.rotation_matrices[i],
                allow_missing_keys=True
            )
            for i in range(self.n_rotations)
        ]
        self.inv_rotations = [
            transform.inverse
            for transform in self.rotations
        ]
        self.inv_hc_rotations = [
            transforms.Affined(
                keys=["image", "label"],
                mode=["bilinear", "nearest"],
                affine=inverse_rotation_matrices[i],
                allow_missing_keys=True
            )
            for i in range(self.n_rotations)
        ]

    def __len__(self):
        return len(self.metadata)

    def get_sample(self, index):

        item = self.metadata[index]
        sample = self.base(deepcopy(item))

        return sample

    def get_rotated_sample(self, index):

        item = self.metadata[index]
        sample = self.base(deepcopy(item))
        rotated_samples = {
            "image": [],
            "label": []
        }
        
        for rotation in self.rotations:
            rotated_sample = self.preprocessing(rotation(sample))
            rotated_samples["image"].append(rotated_sample["image"])
            rotated_samples["label"].append(rotated_sample["label"])
        
        sample = self.preprocessing(sample)
        rotated_samples["image"] = torch.stack(rotated_samples["image"])
        rotated_samples["label"] = torch.stack(rotated_samples["label"])

        return sample, rotated_samples

    def get_double_rotated_sample(self, index):

        item = self.metadata[index]
        sample = self.base(deepcopy(item))
        double_rotated_samples = {
            "image": [],
            "label": []
        }
        rotated_samples = deepcopy(double_rotated_samples)
        
        for rotation, inv_rotation in tqdm(zip(self.rotations, self.inv_rotations), total=self.n_rotations):
            rotated_sample = rotation(sample)
            double_rotated_sample = inv_rotation(rotated_sample)
            processed_rotated = self.preprocessing(rotated_sample)
            processed_double_rotated = self.preprocessing(double_rotated_sample)
            rotated_samples["image"].append(processed_rotated["image"])
            rotated_samples["label"].append(processed_rotated["label"])
            double_rotated_samples["image"].append(processed_double_rotated["image"])
            double_rotated_samples["label"].append(processed_double_rotated["label"])
        
        sample = self.preprocessing(sample)
        rotated_samples["image"] = torch.stack(rotated_samples["image"])
        rotated_samples["label"] = torch.stack(rotated_samples["label"])
        double_rotated_samples["image"] = torch.stack(double_rotated_samples["image"])
        double_rotated_samples["label"] = torch.stack(double_rotated_samples["label"])

        return sample, rotated_samples, double_rotated_samples


    def rotate_sample(self, sample, i):

        return self.rotations[i](sample)
    
    def inverse_rotate_sample(self, sample, i):

        return self.inv_rotations[i](sample)
    
    def inverse_hc_rotate_sample(self, sample, i):

        return self.inv_hc_rotations[i](sample)
        

def harmonic_mean(value1, value2):
    return 2 * (1 / value1 + 1 / value2) ** -1


def find_slices_with_largest_errors(seg1, seg2, ground_truth, axis=2):

    dice = Dice(num_classes=14, average="macro")
    num_slices = seg1.size(axis)
    error_scores = []
    
    for i in range(int(0.2 * num_slices), int(0.8 * num_slices) + 1, 10):

        slice1 = seg1.select(axis, i)
        slice2 = seg2.select(axis, i)
        slice_gt = ground_truth.select(axis, i)
        
        dice1 = dice(slice1, slice2)
        dice2 = dice(slice2, slice_gt)
        
        error_score = dice1
        if 0.5 < dice2:
            error_scores.append((i, error_score))
    
    error_scores.sort(key=lambda x: x[1])
    worst_slices = [index for index, _ in error_scores[:4]]
    
    return worst_slices


def find_two_largest_connected_components(binary_map):
    
    bin_mat = binary_map.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    bin_mat = cv2.Mat(bin_mat)
    bin_mat = cv2.convertScaleAbs(bin_mat)
    # bin_mat = cv2.dilate(bin_mat, np.ones((3, 3), np.uint8), iterations=3)
    _, labels_im = cv2.connectedComponents(bin_mat)
    sizes = np.bincount(labels_im.flatten())

    sizes[0] = 0
    largest_labels = np.argsort(sizes)[-2:]
    # if sizes[largest_labels[0]] < 100:
        # largest_labels = largest_labels[[1]]

    boxes = []
    for label in largest_labels:
        
        component = (labels_im == label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            boxes.append((x, y, w, h))

    return boxes


def compute_difference_map(slice1, slice2):
    return (slice1 != slice2).to(torch.int)


def merge_rectangles(rectangles):
    if len(rectangles) == 1:
        return rectangles
    
    rect1, rect2 = rectangles
    if not rectangles_intersect(rect1, rect2):
        return rectangles
    if rect1 == (0, 0, 281, 281):
        return [rect2]
    if rect2 == (0, 0, 281, 281):
        return [rect1]

    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1 + w1, x2 + w2)
    y_max = max(y1 + h1, y2 + h2)

    merged_w = x_max - x_min
    merged_h = y_max - y_min

    return [(x_min, y_min, merged_w, merged_h)]


def get_rectangle(class_map1, class_map2):

    bin_mask = compute_difference_map(class_map1, class_map2)
    boxes = find_two_largest_connected_components(bin_mask)
    boxes = merge_rectangles(boxes)
    rectangles = [
        patches.Rectangle(
            (x-5, y-5),
            w+10,
            h+10,
            linewidth=3,
            edgecolor='red',
            facecolor='none',
            alpha=0.5
        )
        for (x, y, w, h) in boxes
    ]

    return rectangles


def rectangles_intersect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    return (x1 - 7 <= x2 <= x1 + w1 + 7 or x2 - 7 <= x1 <= x2 + w2 + 7) and (y1 - 7 <= y2 <= y1 + h1 + 7 or y2 - 7 <= y1 <= y2 + h2 + 7)
    # return (x1 <= x2 <= x1 + w1 or x2 <= x1 <= x2 + w2) and (y1 <= y2 <= y1 + h1 or y2 <= y1 <= y2 + h2)
    # return False


def visualize_slices(path_to_output, run_name, seg1, seg2, bg_seg1, bg_seg2, ground_truth, slice_indices, axis=2):

    num_slices = len(slice_indices)    
    path_to_save = path_to_output.joinpath(f"slices_comparison_{run_name}.png")

    n_classes = 13
    cmap = plt.get_cmap('hsv', n_classes)
    colors = cmap(np.arange(n_classes)+1)
    classes = [
        "селезенка", 
        "правая почка", 
        "левая почка", 
        "желчный пузырь", 
        "пищевод", 
        "печень", 
        "желудок", 
        "аорта", 
        "нижняя полая вена", 
        "воротная вена и селезеночная вена", 
        "поджелудочная железа", 
        "правая надпочечная железа", 
        "левая надпочечная железа"
    ]
    legend_patches = [Patch(color=colors[i], label=classes[i], alpha=0.5) for i in range(n_classes)]
    non_empty_indices = []
    for slice_idx in slice_indices:
        if ground_truth.select(axis, slice_idx).sum() > np.arange(13).sum():
            non_empty_indices.append(slice_idx)
    n = len(non_empty_indices)
    fig, axs = plt.subplots(n, 3, figsize=(16, 5 * n))
    
    for i, slice_idx in enumerate(non_empty_indices):

        slice1 = bg_seg1.select(axis, slice_idx)
        slice2 = bg_seg2.select(axis, slice_idx)
        slice_gt = ground_truth.select(axis, slice_idx)
        rectangles = get_rectangle(seg1.select(axis, slice_idx), seg2.select(axis, slice_idx))
        
        axs[i, 0].imshow(slice_gt.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axs[i, 0].set_title(f'Истинно верная карта классов - Срез {slice_idx}')
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        
        axs[i, 1].imshow(slice2.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axs[i, 1].set_title(f'Сегментация на объеме - Срез {slice_idx}')
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])

        axs[i, 2].imshow(slice1.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axs[i, 2].set_title(f'Сегментация на повернутом объеме - Срез {slice_idx}')
        axs[i, 2].set_xticks([])
        axs[i, 2].set_yticks([])

        for rectangle in rectangles:
            axs[i, 0].add_patch(deepcopy(rectangle))
            axs[i, 1].add_patch(deepcopy(rectangle))
            axs[i, 2].add_patch(deepcopy(rectangle))
    
    legend = axs[0, 2].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);
    legend.get_frame().set_facecolor('lightgray')
    plt.tight_layout()
    plt.savefig(path_to_save)

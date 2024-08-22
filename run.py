from utils import Config, CustomDataset, find_slices_with_largest_errors, visualize_slices, calc_scale_coef
from monai.data import decollate_batch, CacheDataset, DataLoader, Dataset
from monai.visualize.utils import matshow3d, blend_images
from lightning.pytorch.callbacks import ModelCheckpoint
from monai.inferers import sliding_window_inference
from lightning.pytorch.loggers import WandbLogger
from monai.data import load_decathlon_datalist
from torchmetrics import MeanAbsoluteError
from monai.transforms import Transform
from monai.config import print_config
from monai.networks.nets import UNETR
from torch.nn import functional as F
from monai.metrics import DiceMetric
from argparse import ArgumentParser
from monai import data, transforms
import matplotlib.pyplot as plt
from lightning import Trainer
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import wandb
import json


parser = ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
args = parser.parse_args()
path_to_config = args.config

print(f"{path_to_config.upper()} IS RUNNING")
config = Config(path_to_config)
config.init()
config.rotation_degrees = np.array(config.rotation_degrees)
config.path_to_output = Path(config.path_to_output)
config.path_to_output.mkdir(exist_ok=True)
config_log = {k: v for k, v in dict(vars(config)).items() if k[:2] != "__"}
run_name = f"{config.image_rotation_mode}_{config.label_rotation_mode}"


model = UNETR(
    in_channels=config.in_channels,
    out_channels=config.out_channels,
    img_size=config.img_size,
    feature_size=config.feature_size,
    hidden_size=config.hidden_size,
    mlp_dim=config.mlp_dim,
    num_heads=config.num_heads,
    pos_embed=config.pos_embed,
    norm_name=config.norm_name,
    conv_block=config.conv_block,
    res_block=config.res_block,
    dropout_rate=config.dropout_rate
)

model_dict = torch.load(config.path_to_model_checkpoint)
model.load_state_dict(model_dict, strict=False)
model = model.eval().to(config.device)


metadata = load_decathlon_datalist(
    config.path_to_meta,
    True,
    "validation",
    base_dir=config.path_to_btcv
)

dataset = CustomDataset(
    metadata=metadata,
    image_rotation_mode=config.image_rotation_mode,
    label_rotation_mode=config.label_rotation_mode,
    rotation_degrees=config.rotation_degrees,
    add_padding=config.add_padding
)

result_json = {
    "Rotation degrees": config.rotation_degrees.tolist(),
    "Image rotation mode": config.image_rotation_mode,
    "Label rotation mode": config.label_rotation_mode
}

to_onehot = transforms.Compose([
    transforms.EnsureType("tensor", device="cpu"),
    transforms.AsDiscrete(to_onehot=14)
])
to_onehot_output = transforms.Compose([
        transforms.EnsureType("tensor", device="cpu"),
        transforms.AsDiscrete(argmax=True, to_onehot=14)
])
to_discrete = transforms.Compose([
    transforms.EnsureType("tensor", device="cpu"),
    transforms.AsDiscrete(argmax=True)
])
mae_metric, mae_values, dice_metrics, dice_output_metrics = MeanAbsoluteError(), [], [], []
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_output_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

for i in tqdm(range(len(dataset))):

    sample, rotated_sample, transformed_sample = dataset.get_double_rotated_sample(i)
    label, transformed_label = to_onehot(deepcopy(sample["label"])), to_onehot(transformed_sample["label"])
    rotated_image = rotated_sample["image"].unsqueeze(0).to(config.device)
    rotated_label = to_onehot(rotated_sample["label"])

    mae_values.append(mae_metric(transformed_sample["image"], sample["image"]).item() * calc_scale_coef(sample["image"], transformed_sample["image"]))
    dice_metric(y_pred=transformed_label, y=label)
    dice_metrics.append(dice_metric.aggregate().item())
    dice_metric.reset()

    if config.evaluate_rotation_effect:
        with torch.no_grad():
            prediction = sliding_window_inference(rotated_image, config.img_size, 4, model, overlap=config.overlap)
            prediction_ = to_onehot_output(deepcopy(prediction[0]))
            dice_output_metric(y_pred=prediction_, y=rotated_label)
            dice_output_metrics.append(dice_output_metric.aggregate().item())
            dice_output_metric.reset()

    if config.generate_images and i == 0:
        if not config.evaluate_rotation_effect:
            with torch.no_grad():
                prediction = sliding_window_inference(rotated_image, config.img_size, 4, model, overlap=config.overlap)
        p_r = to_discrete(prediction[0])
        p_r = dataset.inv_label_rotation({"label": p_r})["label"]
        x_r = transformed_sample["image"]
        x = sample["image"].to(config.device)
        with torch.no_grad():
            p = to_discrete(
                sliding_window_inference(x.unsqueeze(0), config.img_size, 4, model, overlap=config.overlap)[0]
            )
        x = x.cpu()
        g = sample["label"]


if config.check_correctness:
    result_json["MAE error on original images"] = np.mean(mae_values)
    result_json["DICE metric on labels"] = np.mean(dice_metrics)
if config.evaluate_rotation_effect:
    result_json["DICE metric on segmentations"] = np.mean(dice_output_metrics)

if config.check_correctness or config.evaluate_rotation_effect:
    with open(config.path_to_output.joinpath(f"metrics_{run_name}.json"), "w") as f:
        json.dump(result_json, f, indent=4)

if config.use_wandb:
    wandb.init(
        project="Diploma",
        name=config.run_name + f"_{config.rotation_degrees}",
        config=config_log
    )
    wandb.log(result_json)
    wandb.finish()


if config.generate_images:
    p_r, p, g = p_r.to(torch.int), p.to(torch.int), g.to(torch.int)
    for i in range(14):
        p_r[0, i, 1, 0] = i
        p[0, i, 1, 0] = i
        g[0, i, 1, 0] = i

    idxs = find_slices_with_largest_errors(p_r, p, g, axis=-1)
    p_x_r = blend_images(x_r, p_r)[:, 100:-100, 100:-100, :]
    p_x = blend_images(x, p)[:, 100:-100, 100:-100, :]
    g_x = blend_images(x, g)[:, 100:-100, 100:-100, :]
    p_r = p_r[:, 100:-100, 100:-100, :]
    p = p[:, 100:-100, 100:-100, :]

    visualize_slices(config.path_to_output, run_name, p_r, p, p_x_r, p_x, g_x, idxs, axis=-1)

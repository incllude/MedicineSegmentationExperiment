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
import gc


parser = ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
args = parser.parse_args()
path_to_config = args.config

print(f"{path_to_config.upper()} IS RUNNING")
config = Config(path_to_config)
config.init()
config.path_to_output = Path(config.path_to_output)
config.path_to_output.mkdir(exist_ok=True)
config_log = {k: v for k, v in dict(vars(config)).items() if k[:2] != "__"}
run_name = f"from_{config.angle_range[0]}_to_{config.angle_range[1]}"


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

border = int(490 * (np.sin((20 + 45) * np.pi / 180) - np.sin(np.pi / 4)))
dataset = CustomDataset(
    metadata=metadata,
    angle_range=config.angle_range,
    n_rotations=config.n_rotations,
    add_padding=config.add_padding,
    border=border
)

result_json = {
    "Rotation degrees range": config.angle_range,
    "Num rotations": config.n_rotations
}

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


for i in range(len(dataset)):
    with torch.no_grad():

        sample = dataset.get_sample(i)
        preprocessed_sample = dataset.preprocessing(sample)
        image = preprocessed_sample["image"]
        label = preprocessed_sample["label"]

        for j in tqdm(range(dataset.n_rotations)):

            rotated_sample = dataset.rotate_sample(sample, j)
            transformed_sample = dataset.inverse_rotate_sample(rotated_sample, j)
            preprocessed_rotated_sample = dataset.preprocessing(rotated_sample)
            preprocessed_transformed_sample = dataset.preprocessing(transformed_sample)
            rotated_image = preprocessed_rotated_sample["image"].unsqueeze(0).to(config.device)
            rotated_label = preprocessed_rotated_sample["label"].unsqueeze(0)
            transformed_image = preprocessed_transformed_sample["image"]
            transformed_label = preprocessed_transformed_sample["label"]

            mae_values.append(mae_metric(image, transformed_image).item() * calc_scale_coef(image, transformed_image))
            dice_metric(y_pred=transformed_label, y=label)

            if config.evaluate_rotation_effect:
                predictions = sliding_window_inference(rotated_image, config.img_size, config.batch_size, model, overlap=config.overlap)
                predictions_ = to_onehot_output(predictions[0]).unsqueeze(0)
                dice_output_metric(y_pred=predictions_, y=rotated_label)
                dice_output_metrics.append((i, j, dice_output_metric.aggregate().item()))
                dice_output_metric.reset()

                predictions = predictions.cpu()
                predictions_ = predictions_.cpu()

            rotated_image = rotated_image.cpu()

        dice_metrics.append(dice_metric.aggregate().item())
        dice_metric.reset()


if config.generate_images:

    dice_output_metrics = sorted(dice_output_metrics, key=lambda x: x[2])
    i, j, median_score = dice_output_metrics[len(dice_output_metrics) // 2]

    sample = dataset.get_sample(i)
    rotated_sample = dataset.rotate_sample(sample, j)
    transformed_sample = dataset.inverse_rotate_sample(rotated_sample, j)
    preprocessed_sample = dataset.preprocessing(sample)
    preprocessed_rotated_sample = dataset.preprocessing(rotated_sample)
    preprocessed_transformed_sample = dataset.preprocessing(transformed_sample)
    rotated_image = preprocessed_rotated_sample["image"].to(config.device)
    transformed_image = preprocessed_transformed_sample["image"]
    x = preprocessed_sample["image"].to(config.device)
    g = to_discrete(preprocessed_sample["label"])

    with torch.no_grad():
        prediction = sliding_window_inference(rotated_image.unsqueeze(0), config.img_size, config.batch_size, model, overlap=config.overlap)
    p_r = to_discrete(prediction[0])
    p_r = dataset.inverse_hc_rotate_sample({"label": p_r}, j)["label"]
    x_r = transformed_image
    with torch.no_grad():
        prediction = sliding_window_inference(x.unsqueeze(0), config.img_size, config.batch_size, model, overlap=config.overlap)
    p = to_discrete(prediction[0])
    x = x.cpu()


dice_output_metrics = list(map(lambda x: (x[2]), dice_output_metrics))


if config.check_correctness:
    result_json["MAPE error on original images"] = np.mean(mae_values)
    result_json["Mean DICE metric on labels"] = np.mean(dice_metrics)
if config.evaluate_rotation_effect:
    result_json["Mean DICE metric on segmentations"] = np.mean(dice_output_metrics)
if config.generate_images:
    result_json["Sample index"] = i
    result_json["Rotation index"] = j
    result_json["Median DICE metric on segmentations"] = median_score


if config.check_correctness or config.evaluate_rotation_effect:
    with open(config.path_to_output.joinpath(f"metrics_{run_name}.json"), "w") as f:
        json.dump(result_json, f, indent=4)

if config.use_wandb:
    wandb.init(
        project="Diploma",
        name=f"metrics_{run_name}",
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

    m = border // 2
    idxs = find_slices_with_largest_errors(p_r, p, g, axis=-1)
    p_x_r = blend_images(x_r, p_r)[:, m:-m, m:-m, :]
    p_x = blend_images(x, p)[:, m:-m, m:-m, :]
    g_x = blend_images(x, g)[:, m:-m, m:-m, :]
    p_r = p_r[:, m:-m, m:-m, :]
    p = p[:, m:-m, m:-m, :]

    visualize_slices(config.path_to_output, run_name, p_r, p, p_x_r, p_x, g_x, idxs, axis=-1)

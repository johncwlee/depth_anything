# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import argparse
from pprint import pprint
import os
import torch
import torch.nn.functional as F
import numpy as np
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics, count_parameters)


@torch.no_grad()
def infer(model, images, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred


@torch.no_grad()
def predict(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()

    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth = sample['image'], sample['depth']
        image, depth = image.cuda(), depth.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        pred = infer(model, image, dataset=sample['dataset'][0], focal=focal)
        
        pred_out = pred.clone()
        if depth.shape[-2:] != pred_out.shape[-2:]:
            pred_out = F.interpolate(
                pred_out, depth.shape[-2:], mode='bilinear', align_corners=True)

        pred_out = pred.clone()
        pred_out = pred_out.squeeze().cpu().numpy()
        pred_out[pred_out < config.min_depth_eval] = config.min_depth_eval
        pred_out[pred_out > config.max_depth_eval] = config.max_depth_eval
        pred_out[np.isinf(pred_out)] = config.max_depth_eval
        pred_out[np.isnan(pred_out)] = config.min_depth_eval

        #? Save depth
        if config.save_images and config.save_dir is not None:
            if sample['dataset'][0] == "semantickitti":
                sequence_id = sample['image_path'][0].split('/')[-3]
                frame_id = sample['image_path'][0].split('/')[-1].split('.')[0]
                out_path = os.path.join(config.save_dir, "sequences",sequence_id)
                
                os.makedirs(out_path, exist_ok=True)
                np.save(os.path.join(out_path, f"{frame_id}.npy"), pred_out)

        metrics.update(compute_metrics(depth, pred, config=config))

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics

def main(config):
    model = build_model(config)
    test_loader = DepthDataLoader(config, 'online_eval').data
    model = model.cuda()
    metrics = predict(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    return metrics


def predict_model(model_name, pretrained_resource, dataset='semantickitti', save_dir='./results', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite_kwargs = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    if dataset == "allo":
        overwrite_kwargs["config_version"] = "allo"
    elif dataset == "STU":
        overwrite_kwargs["config_version"] = "stu"
    elif dataset == "STU-Mix":
        overwrite_kwargs["config_version"] = "stu-mix"
    elif dataset == "semantickitti":
        overwrite_kwargs["config_version"] = "semantickitti"
    else:
        overwrite_kwargs["config_version"] = None
    
    config = get_config(model_name, "eval", dataset, **overwrite_kwargs)
    config.save_dir = save_dir
    
    pprint(config)
    print(f"Evaluating and saving predictions of {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        default='kitti', help="Dataset to evaluate on")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--save_images", type=bool, default=False)

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    predict_model(args.model, pretrained_resource=args.pretrained_resource,
                dataset=args.dataset, save_dir=args.save_dir, save_images=args.save_images, **overwrite_kwargs)

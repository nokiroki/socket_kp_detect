import datetime
import errno
import os
import time
from collections import defaultdict, deque
import math

import torch
import torch.distributed as dist
import albumentations as A

from classes.metric_logger import MetricLogger
from classes.smooth_value import SmoothedValue

def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model,
                    optimizer,
                    data_loader,
                    device,
                    epoch,
                    print_freq,
                    scaler=None):
    model.train()
    metricLogger = MetricLogger(delimiter="  ")
    metricLogger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metricLogger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metricLogger.update(loss=losses_reduced, **loss_dict_reduced)
        metricLogger.update(lr=optimizer.param_groups[0]["lr"])

    return metricLogger


def train_transform():
    return A.Compose([
        A.Sequential([
            A.augmentations.Affine(translate_px={'x':[-55,55],'y':[-25,25]}, interpolation=1, mode=4, p=1.),
            A.Rotate (limit=15, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.95),
            A.RandomBrightnessContrast(brightness_limit=0.01, contrast_limit=0.01, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'),
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more here https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    #print(dir(coco))
    iou_types = _get_iou_types(model)
    #print('iou_types', iou_types)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    #print(dir(coco))
    #print(coco.getCatIds())

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    
    
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True
    
    
    
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
    
    
def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


if __name__ == '__main__':
    print('ty pidor')
    print(os.listdir())
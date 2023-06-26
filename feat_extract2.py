"""
Run evaluation on a trained model to get mAP and class wise AP.

USAGE:
python eval.py --config data_configs/voc.yaml --weights outputs/training/fasterrcnn_convnext_small_voc_15e_noaug/best_model.pth --model fasterrcnn_convnext_small
"""
from datasets import (
    create_valid_dataset, create_valid_loader, create_train_dataset, create_train_loader
)
from models.create_fasterrcnn_model import create_model
from torch_utils import utils
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
from tqdm import tqdm

import torch
import argparse
import yaml
import torchvision
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', 
        default='data_configs/test_image_config.yaml',
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', 
        default='fasterrcnn_resnet50_fpn',
        help='name of the model'
    )
    parser.add_argument(
        '-mw', '--weights', 
        default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-ims', '--imgsz', 
        default=640, 
        type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-w', '--workers', default=4, type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch', 
        default=8, 
        type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='show class-wise mAP'
    )
    parser.add_argument(
        '-st', '--square-training',
        dest='square_training',
        action='store_true',
        help='Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.'
    )
    args = vars(parser.parse_args())

    # Load the data configurations
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)

    # Validation settings and constants.
    try: # Use test images if present.
        VALID_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    except: # Else use the validation images.
        VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    TRAIN_DIR_IMAGES = data_configs['TRAIN_DIR_IMAGES']
    TRAIN_DIR_LABELS = data_configs['TRAIN_DIR_LABELS']
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    BATCH_SIZE = args['batch']

    # Model configurations
    IMAGE_SIZE = args['imgsz']

    # Load the pretrained model
    create_model = create_model[args['model']]
    if args['weights'] is None:
        try:
            model, coco_model = create_model(num_classes=NUM_CLASSES, coco_model=True)
        except:
            model = create_model(num_classes=NUM_CLASSES, coco_model=True)
        if coco_model:
            COCO_91_CLASSES = data_configs['COCO_91_CLASSES']
            valid_dataset = create_valid_dataset(
                VALID_DIR_IMAGES, 
                VALID_DIR_LABELS, 
                IMAGE_SIZE, 
                COCO_91_CLASSES, 
                square_training=args['square_training']
            )
            train_dataset = create_train_dataset(
            TRAIN_DIR_IMAGES,
            TRAIN_DIR_LABELS,
            IMAGE_SIZE,
            CLASSES,
            square_training=args['square_training']
        )

    # Load weights.
    if args['weights'] is not None:
        model = create_model(num_classes=NUM_CLASSES, coco_model=False)
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        valid_dataset = create_valid_dataset(
            VALID_DIR_IMAGES, 
            VALID_DIR_LABELS, 
            IMAGE_SIZE, 
            CLASSES,
            square_training=args['square_training']
        )
        train_dataset = create_train_dataset(
            TRAIN_DIR_IMAGES,
            TRAIN_DIR_LABELS,
            IMAGE_SIZE,
            CLASSES,
            square_training=args['square_training']
        )
            
    model.to(DEVICE).eval()
    
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)
    train_loader = create_train_loader(train_dataset, BATCH_SIZE, NUM_WORKERS)

    @torch.inference_mode()
    def evaluate(
        model, 
        data_loader, 
        device, 
        out_dir=None,
        classes=None,
        colors=None
    ):

        features = []
        def save_features(mod, inp, outp):
          features.append(outp)
          torch.save(features, 'features_fc7.pt')

        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"

        target = []
        preds = []
        features = []  # List to store the extracted features

        # layer_to_hook = 'roi_heads'
        layer_to_hook = 'roi_heads.box_head.fc7'
        for name, layer in model.named_modules():
          if name == layer_to_hook:
            layer.register_forward_hook(save_features)
                
        counter = 0
        for images, targets in tqdm(metric_logger.log_every(data_loader, 100, header), total=len(data_loader)):
            counter += 1
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            with torch.no_grad():
                outputs = model(images)

                # layer_to_hook = 'roi_heads'
                # for name, layer in model.named_modules():
                #     if name == layer_to_hook:
                #         layer.register_forward_hook(save_features)
                # features_dict = model.backbone(images)  # Extract features from backbone
                # features.append(features_dict)
                # Extract image features before ROI pooling
                # roi_features = model.roi_heads.box_roi_pool(
                #     proposals, rpn_features
                # )

            #####################################
            for i in range(len(images)):
                true_dict = dict()
                preds_dict = dict()
                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()
                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                preds.append(preds_dict)
                target.append(true_dict)
                
                # Extract image features
                # image_features = model.roi_heads.box_roi_pool(
                #     outputs[i]['boxes'], outputs[i]['features']
                # )
                # features.append(image_features)
                # features.append(roi_features[i].detach().cpu())
            #####################################

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # gather the stats from all processes
        print(features)
        metric_logger.synchronize_between_processes()
        torch.set_num_threads(n_threads)
        metric = MeanAveragePrecision(class_metrics=args['verbose'])
        metric.update(preds, target)
        metric_summary = metric.compute()
        return metric_summary#, features

    stats = evaluate(
        model, 
        train_loader, 
        device=DEVICE,
        classes=CLASSES,
    )
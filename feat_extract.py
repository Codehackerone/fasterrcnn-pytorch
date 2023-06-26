"""
Run evaluation on a trained model to get mAP and class wise AP.

USAGE:
python feat_extract.py --config data_configs/voc.yaml --weights outputs/training/fasterrcnn_convnext_small_voc_15e_noaug/best_model.pth --model fasterrcnn_convnext_small
"""
from datasets import (
    create_train_dataset, create_train_loader
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
        '--feature', 
        default='features.pt',
        help='(optional) path to the where features will be saved'
    )
    parser.add_argument(
        '--layer', 
        default=None,
        help='layer to hook'
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
    # try: # Use test images if present.
    #     VALID_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
    #     VALID_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    # except: # Else use the validation images.
    #     VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
    #     VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    
    TRAIN_DIR_IMAGES = data_configs['TRAIN_DIR_IMAGES']
    TRAIN_DIR_LABELS = data_configs['TRAIN_DIR_LABELS']
    
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    BATCH_SIZE = args['batch']

    # Model configurations
    IMAGE_SIZE = args['imgsz']
    FEATURE_FILE_PATH = args['feature']
    LAYER_TO_HOOK = args['layer']

    # Load the pretrained model
    create_model = create_model[args['model']]
    if args['weights'] is None:
        try:
            model, coco_model = create_model(num_classes=NUM_CLASSES, coco_model=True)
        except:
            model = create_model(num_classes=NUM_CLASSES, coco_model=True)
        if coco_model:
            COCO_91_CLASSES = data_configs['COCO_91_CLASSES']
            train_dataset = create_train_dataset(
                TRAIN_DIR_IMAGES, 
                TRAIN_DIR_LABELS, 
                IMAGE_SIZE, 
                COCO_91_CLASSES, 
                square_training=args['square_training']
            )

    # Load weights.
    if args['weights'] is not None:
        model = create_model(num_classes=NUM_CLASSES, coco_model=False)
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        train_dataset = create_train_dataset(
            TRAIN_DIR_IMAGES, 
            TRAIN_DIR_LABELS, 
            IMAGE_SIZE, 
            CLASSES,
            square_training=args['square_training']
        )
    model.to(DEVICE).eval()
    
    train_loader = create_train_loader(train_dataset, BATCH_SIZE, NUM_WORKERS)

    @torch.inference_mode()
    def extract(
        model,        
        data_loader, 
        device, 
        layer_to_hook = 'rpn',
        out_dir=None,
        classes=None,
        colors=None
    ):

        features = []
        def save_features(mod, inp, outp):
          features.append(outp)
          torch.save(features, FEATURE_FILE_PATH)

        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        model.eval()        

        for name, layer in model.named_modules():
          if name == layer_to_hook:
            layer.register_forward_hook(save_features)                      

    extract(
      model,         
      train_loader, 
      device=DEVICE,
      layer_to_hook='rpn',
      classes=CLASSES,
    )  
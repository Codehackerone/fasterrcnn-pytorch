from models.create_fasterrcnn_model import create_model
import yaml
import torch

config_path = '/content/fasterrcnn-pytorch/data_configs/custom_data.yaml'
weight_path = '/content/fasterrcnn-pytorch/outputs/training/custom_training/best_model.pth'
create_model = create_model['fasterrcnn_resnet50_fpn']

with open(config_path) as file:
  data_configs = yaml.safe_load(file)

NUM_CLASSES = data_configs['NC']
CLASSES = data_configs['CLASSES']
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = create_model(num_classes=NUM_CLASSES, coco_model=False)
checkpoint = torch.load(weight_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# Target
target_layer = model.roi_heads.box_head.fc7

# Create a placeholder variable to store the extracted output
output_features = None

# Define the forward hook function to capture the output of the target layer
def hook(module, input, output):
  global output_features
  output_features = output

# Register the forward hook to the target layer
hook_handle = target_layer.register_forward_hook(hook)

# Prepare and pass the input image through the model
image = torch.randn(1, 3, 224, 224).to(DEVICE)  # Replace with your input image tensor
with torch.no_grad():
    model(image)

# Remove the hook after obtaining the output
hook_handle.remove()

# Access the output features
print("Output Features Shape:", output_features.shape)

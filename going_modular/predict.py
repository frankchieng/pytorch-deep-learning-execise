"""
Predict a PyTorch image using device-agnostic code.
"""
import argparse
import pathlib
from pathlib import Path
import model_builder
import torch
from torchvision import transforms
import torchvision
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"

parser = argparse.ArgumentParser(description="this is the inference prediction command line parameters explanation")
parser.add_argument('--predict_image', type=pathlib.Path, required=True, help="image prediction filepath")
args = parser.parse_args()

custom_image_path = data_path / args.predict_image

MODEL_PATH = Path("models")
MODEL_NAME = "05_going_modular_script_mode_tinyvgg_model_with_cli_params.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


loaded_model = model_builder.TinyVGG(input_shape=3, 
                                    hidden_units=20, # try changing this to 128 and seeing what happens 
                                    output_shape=3) 

# Load in the saved state_dict()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send model to GPU
loaded_model = loaded_model.to(device)

# Create transform pipleine to resize image
custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
])


def pred_image(model: torch.nn.Module, 
                        image_path: str, 
                        class_names: List[str] = None, 
                        transform=None,
                        device: torch.device = device):
    """Makes a prediction on a target image and plots the image with its prediction."""
    
    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    
    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255. 
    
    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)
    
    # 4. Make sure the model is on the target device
    model.to(device)
    
    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
    
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))
        
    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    # 8. prediction probability

    if class_names:
        result = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else: 
        result = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    return result

# Use ImageFolder to create dataset(s)
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=custom_image_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)
# Get class names as a list
class_names = train_data.classes

# Pred on our custom image
pred_result = pred_image(model=loaded_model,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform,
                    device=device)
print(pred_result)

import torch
from PIL import Image
from torchvision import transforms
from processor import get_model
import yaml
import argparse
import torchvision

# Argument parsing
parser = argparse.ArgumentParser(description='Check similarity between two images using a trained re-ID model')
parser.add_argument('--img1', required=True, help='Path to the first image')
parser.add_argument('--img2', required=True, help='Path to the second image')
parser.add_argument('--config', required=True, help='Path to config YAML file')
parser.add_argument('--weights', required=True, help='Path to trained model weights (.pth)')
parser.add_argument('--metric', choices=['euclidean', 'cosine'], default='euclidean', help='Similarity metric to use')
parser.add_argument('--threshold', type=float, default=None, help='Threshold for similarity decision')
args = parser.parse_args()

# 1. Load config and model
with open(args.config, 'r') as f:
    data = yaml.safe_load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(data, device)
model.load_state_dict(torch.load(args.weights, map_location=device))  # adjust path
model.eval()

# 2. Define preprocessing (same as in main.py)
preprocess = transforms.Compose([
    transforms.Resize((data['y_length'], data['x_length']), antialias=True),
    transforms.Normalize(data['n_mean'], data['n_std']),
])

# 3. Load and preprocess images
def load_image(path):
    img = torchvision.io.read_image(path)  # returns tensor [C, H, W], uint8
    img = img.type(torch.FloatTensor) / 255.0
    img = preprocess(img)
    return img.unsqueeze(0)  # add batch dimension

img1 = load_image(args.img1).to(device)
img2 = load_image(args.img2).to(device)

# 4. Extract features
with torch.no_grad():
    # Dummy cam/view values if needed
    cam = torch.zeros(1, dtype=torch.long).to(device)
    view = torch.zeros(1, dtype=torch.long).to(device)
    _, _, ffs1, _ = model(img1, cam, view)
    _, _, ffs2, _ = model(img2, cam, view)
    # Aggregate features as in test_epoch
    feat1 = torch.cat([torch.nn.functional.normalize(f) for f in ffs1], 1)
    feat2 = torch.cat([torch.nn.functional.normalize(f) for f in ffs2], 1)
    # Optionally normalize the final vector
    feat1 = torch.nn.functional.normalize(feat1)
    feat2 = torch.nn.functional.normalize(feat2)

# 5. Compute similarity
if args.metric == 'euclidean':
    sim_value = torch.norm(feat1 - feat2, p=2).item()
    print(f"Euclidean distance: {sim_value}")
    if args.threshold is not None:
        if sim_value < args.threshold:
            print(f"Images are considered SIMILAR (distance < {args.threshold})")
        else:
            print(f"Images are considered NOT similar (distance >= {args.threshold})")
elif args.metric == 'cosine':
    sim_value = torch.nn.functional.cosine_similarity(feat1, feat2).item()
    print(f"Cosine similarity: {sim_value}")
    if args.threshold is not None:
        if sim_value > args.threshold:
            print(f"Images are considered SIMILAR (similarity > {args.threshold})")
        else:
            print(f"Images are considered NOT similar (similarity <= {args.threshold})")
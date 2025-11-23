
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import glob
import sys
sys.path.append('image2sphere')
from src.predictor import I2S
from src.visualizations import plot_predictions

# download the checkpoint
%%capture

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dWlvGp1QY3esAqZgOnPikR6TpJmBUngd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dWlvGp1QY3esAqZgOnPikR6TpJmBUngd" -O pascal3d_checkpoint.pt && rm -rf /tmp/cookies.txt

# use untrained resnet101 to avoid downloading weights that will be overwritten
# this will take a minute to generate the Wigner-D matrices for the output grid
model = I2S(encoder='resnet101', eval_grid_rec_level=3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(
    torch.load('pascal3d_checkpoint.pt', map_location=device)['model_state_dict'],
)
model.eval()
'''
Preprocess Images into correct format
Code adapted from https://github.com/tpark94/spnv2/blob/main/tools/preprocess.py
'''
import json
import os
import cv2
import tqdm
import numpy as np
import torch #added for sam2

datadir = '/home/galin.j/speedplus/speedplusv2/synthetic'
# Read labels from JSON file
jsonfile = '/home/galin.j/speedplus/speedplusv2/synthetic/validation.json'
print(f'Reading JSON file from {jsonfile}...')
with open(jsonfile, 'r') as f:
    labels = json.load(f) # list

domain = ''
split = 'validation.json'
outdir = os.path.join(datadir, domain, 'labels')
if not os.path.exists(outdir): os.makedirs(outdir)
csvfile = os.path.join(outdir, split.replace('json', 'csv'))
print(f'Label CSV file will be saved to {csvfile}')

# Where to save resized image?
imagedir = os.path.join(datadir, domain,
        f'images_{768}x{512}_RGB')
if not os.path.exists(imagedir): os.makedirs(imagedir)
print(f'Resized images will be saved to {imagedir}')


# if args.load_masks:
#     maskdir = os.path.join(datadir, domain,
#         f'masks_{int(cfg.DATASET.INPUT_SIZE[0]/cfg.DATASET.OUTPUT_SIZE[0])}x{int(cfg.DATASET.INPUT_SIZE[1]/cfg.DATASET.OUTPUT_SIZE[0])}')
#     if not os.path.exists(maskdir): os.makedirs(maskdir)
#     print(f'Resized masks will be saved to {maskdir}')

# Open
csv = open(csvfile, 'w')

for idx in tqdm.tqdm(range(len(labels))):

    # ---------- Read image & resize & save
    filename = labels[idx]['filename']
    image    = cv2.imread(os.path.join(datadir, 'images', filename), cv2.IMREAD_COLOR)
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image    = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image    = cv2.resize(image, [768, 512])
    cv2.imwrite(os.path.join(imagedir, filename), image)

    # ---------- Read mask & resize & save
    # if args.load_masks:
    #     mask     = cv2.imread(os.path.join(datadir, domain, 'masks', filename), cv2.IMREAD_GRAYSCALE)
    #     mask_out = cv2.resize(mask, [int(s / cfg.DATASET.OUTPUT_SIZE[0]) for s in cfg.DATASET.INPUT_SIZE])
    #     cv2.imwrite(os.path.join(maskdir, filename), mask_out)

    # ---------- Read labels
    q_vbs2tango = np.array(labels[idx]['q_vbs2tango_true'], dtype=np.float32)
    r_Vo2To_vbs = np.array(labels[idx]['r_Vo2To_vbs_true'], dtype=np.float32)

    # ---------- Bounding box labels
    # If masks are available, get them from masks
    # If not, use keypoints instead
    # if args.load_masks:
    #     seg  = np.where(mask > 0)
    #     xmin = np.min(seg[1]) / camera['Nu']
    #     ymin = np.min(seg[0]) / camera['Nv']
    #     xmax = np.max(seg[1]) / camera['Nu']
    #     ymax = np.max(seg[0]) / camera['Nv']


    # CSV row
    row = [filename]

    row = row + q_vbs2tango.tolist() \
          + r_Vo2To_vbs.tolist() \
          # + [xmin, ymin, xmax, ymax]
    row = ', '.join([str(e) for e in row])

    # Write
    csv.write(row + '\n')

csv.close()

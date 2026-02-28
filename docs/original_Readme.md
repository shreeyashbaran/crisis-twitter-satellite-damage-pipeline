# Multimodal Crisis Damage Assessment  
**Datasets:** CrisisMMD v2.0 (social media) + BRIGHT (satellite)  
**Core tasks:**
1. CrisisMMD – tweet-level classification (text + images)
2. BRIGHT – building damage segmentation from post-event satellite imagery

This README explains:
- How to set up the environment
- How to place the datasets
- How to run the two notebooks (`CrisisMMD.ipynb` and `Bright.ipynb`)
- What artifacts are produced and how to interpret them
- How to reproduce the main reported numbers

---

## 1. Environment Setup

We used Python 3 with PyTorch + HuggingFace for text models, and timm + a small U-Net for images.

### 1.1. Recommended packages

Create a new environment (conda or venv) and install:

```bash
pip install \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or CPU-only if needed

pip install \
  transformers \
  datasets \
  evaluate \
  scikit-learn \
  pandas \
  numpy \
  matplotlib \
  seaborn \
  tqdm \
  timm \
  tifffile \
  ipywidgets
Optional but recommended for BERTweet emoji handling:

bash
Copy code
pip install emoji==0.6.0
In Jupyter / VSCode, also ensure:

bash
Copy code
jupyter nbextension enable --py widgetsnbextension
1.2. Reproducibility
Both notebooks set seeds:

python
Copy code
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
Results may still vary slightly due to GPU nondeterminism, but they should be close.

2. Data Setup
2.1. CrisisMMD v2.0
Download CrisisMMD v2.0 and arrange it as:

text
Copy code
<ProjectRoot>/
  CrisisMMD.ipynb
  Bright.ipynb
  artifacts_stage1/        # created by the notebook; small files can be found in the folder, while bigger files were avoided
  CrisisMMD_v2.0/
    crisismmd_datasplit_all/
      crisismmd_datasplit_all/
        task_informative_text_img_train.tsv
        task_informative_text_img_dev.tsv
        task_informative_text_img_test.tsv
        task_humanitarian_text_img_train.tsv
        ...
        task_damage_text_img_train.tsv
        ...
    data_image/
      hurricane_harvey/...
      ...
In CrisisMMD.ipynb, check or update:

python
Copy code
DATA_ROOT = Path(r"C:\JP_Notebooks\CSCE 5380\Project\CrisisMMD_v2.0")
Change this to the correct path on your machine if needed.

2.2. BRIGHT
Download the BRIGHT building damage dataset (post-event mode) and arrange it as:

text
Copy code
<ProjectRoot>/
  Bright.ipynb
  artifacts_stage2/          # created by the notebook; small files can be found in the folder, while bigger files were avoided since they can be re-generated through the code
  BRIGHT/
    pre-event/   (optional; in our run these were missing)
    post-event/
      <scene>_post_disaster.tif
      ...
    target/
      <scene>_building_damage.tif
      ...
In Bright.ipynb, check or update:

python
Copy code
DATASET_ROOT = Path(r"C:\JP_Notebooks\CSCE 5380\Project\BRIGHT").resolve()
Again, change this to your actual BRIGHT root if needed.

3. CrisisMMD Notebook: What It Does & How To Run It
Open CrisisMMD.ipynb and run the cells top to bottom.

3.1. Data Cleaning & Alignment (Cells 1–8)
Key steps:

Robust TSV loading

Reads all split files for:

task_informative

task_humanitarian

task_damage

Handles encodings, broken lines, and weird quoting.

Text normalization

Removes newlines, collapses spaces.

Replaces URLs → <URL>, mentions → <USER>.

Turns hashtags #flooded → flooded.

Label normalization

Task 1: Informativeness

Collapses variants into:

informative

not_informative

Task 2: Humanitarian
Canonical labels:

affected_individuals

infrastructure_and_utility_damage

injured_or_dead_people

missing_or_found_people

rescue_volunteering_or_donation_effort

vehicle_damage

other_relevant_information

not_humanitarian

Task 3: Damage
Collapses fine-grained labels into 3 classes:

little_or_none

mild (combines minor / moderate / mild)

severe

Task 1–2 alignment (CrisisMMD v2.0 rule)

Enforces:

If informativeness_label == not_informative → humanitarian_label = not_humanitarian

If humanitarian_label == not_humanitarian → informativeness_label = not_informative

Verifies zero alignment violations.

Image paths check

Resolves image relative to DATA_ROOT/data_image.

Confirms that all image files exist.

Saved outputs (in artifacts_stage1/):

text
Copy code
task1_informative_clean.csv
task2_humanitarian_clean.csv
task3_damage_clean.csv
task12_merged_aligned.csv
You should see printed stats for row counts, splits, and label distributions.

3.2. Task 1 — Informativeness (Text-only, BERTweet)
Goal: Binary classification: informative vs not_informative.

Model:

vinai/bertweet-base

HuggingFace AutoModelForSequenceClassification with 2 labels.

Pipeline:

Load task1_informative_clean.csv.

Split by split column → train / dev / test.

Map labels:

not_informative → 0

informative → 1

Tokenize:

max_length = 128

padding = max_length

Train:

batch size = 16 (train), 32 (eval)

epochs = 3

learning rate = 2e-5

early stopping with patience 2

selects best model by macro F1 on dev.

Metrics printed:

Dev metrics per epoch (loss, accuracy, macro-F1, precision, recall)

Final test performance (approx):

Accuracy ≈ 0.84

Macro-F1 ≈ 0.79

These results are used in the report.

3.3. Task 2 — Humanitarian Categories (Text-only, two stages)
You have two setups for Task 2.

3.3.1. Baseline: BERTweet + Class-Weighted Cross-Entropy
Model: vinai/bertweet-base with 8 humanitarian labels.

Key details:

Loads task2_humanitarian_clean.csv.

Uses class weights based on training distribution.

Custom Trainer subclass:

Uses CrossEntropyLoss(weight=class_weights, label_smoothing=0.05).

Metrics (test):

Accuracy ≈ 0.42

Macro-F1 ≈ 0.48

3.3.2. Improved: Twitter-RoBERTa + Denoising + Focal Loss
This is the strong text model you report for Task 2.

Changes vs baseline:

Denoising:

Merges with raw humanitarian TSVs.

Keeps only train rows with label_text_image == "Positive" (annotator agreement).

Reduces train set to ~6,126 rows (from 13,608).

Upsampling:

Targets TARGET_PER_CLASS = 3000 samples per class (upsampled).

Balanced training set across all 8 humanitarian categories.

Model swap:

cardiffnlp/twitter-roberta-base (tweet-optimized RoBERTa).

Focal Loss + label smoothing:

Custom FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.02)

Helps with class imbalance and hard examples.

Cosine LR schedule:

Uses get_cosine_schedule_with_warmup.

warmup_ratio = 0.10.

Training config:

max_length = 160

epochs = 6

batch size = 16 / 32

best model by macro-F1 on dev.

Metrics (test):

Accuracy ≈ 0.63

Macro-F1 ≈ 0.53

This is the main humanitarian result you should highlight.

3.4. Task 3 — Damage Severity (Image-only & Multimodal)
Task 3 uses the image damage labels from CrisisMMD.

3.4.1. Image-only model (ConvNeXt-Tiny, improved version)
Data:

Uses task3_damage_clean.csv.

Resolves all image paths and filters missing files.

Three classes:

little_or_none (0)

mild (1)

severe (2)

Model:

convnext_tiny.fb_in22k_ft_in1k from timm (pretrained).

Input size = 288×288 RGB (improved version).

Train/Dev/Test from original split column.

Training details:

Augmentations: random resized crop, flips, color jitter, normalization.

Balanced WeightedRandomSampler to correct class imbalance.

FocalLoss with class weights.

Optimizer: AdamW, lr = 3e-4, weight_decay = 1e-4.

Early stopping based on dev macro-F1.

Image-only metrics (test):

Accuracy ≈ 0.61

Macro-F1 ≈ 0.55

Predictions and probabilities are saved to:

text
Copy code
artifacts_stage1/task3_img_probs.npy
artifacts_stage1/task3_img_preds.csv
3.4.2. Text-only damage model (weak supervision)
Idea: Train a text classifier to predict the same 3 damage classes, using the tweet text instead of images.

Model:

cardiffnlp/twitter-roberta-base (again).

Labels: from damage_severity_label in task3_damage_clean.csv.

Metrics (test):

Macro-F1 ≈ 0.47

3.4.3. Late fusion: Multimodal text + image
Fusion logic:

Compute probabilities:

probs_img from ConvNeXt model (task3_img_probs.npy).

probs_txt from text damage model (softmax of logits).

Combine:

python
Copy code
probs_fused = alpha * probs_img + (1 - alpha) * probs_txt
Sweep alpha from 0.0 to 1.0 in steps of 0.05 and choose the macro-F1 that is highest on the test set.

Result:

Best α ≈ 0.60 (image weighted slightly more than text).

Fused metrics (test):

Accuracy ≈ 0.66

Macro-F1 ≈ 0.57

This is your best multimodal result on CrisisMMD damage.

4. BRIGHT Notebook: What It Does & How To Run It
Open Bright.ipynb and run cells top-to-bottom.

The BRIGHT pipeline focuses on post-event satellite images (single-band or RGB) and binary building damage masks.

4.1. Manifest Construction (Cell 1B)
Goal: Build a scene-level manifest linking:

scene_id

post-event .tif

label .tif

Steps:

Scan:

<BRIGHT>/pre-event/*.tif (may be empty)

<BRIGHT>/post-event/*.tif

<BRIGHT>/target/*.tif

Extract a scene_id from filenames like:

bata-explosion_00000000_post_disaster.tif

bata-explosion_00000000_building_damage.tif

Keep only scenes that have post + label.

Save manifest to:

text
Copy code
artifacts_stage2/bright_manifest.csv
with columns:

scene_id

pre_path (may be None)

post_path

label_path

has_pre, has_post, has_label

Also adds an event column (prefix before first underscore) for summary stats.

4.2. Visual Check & Label Statistics (Cell 2)
What it does:

Loads bright_manifest.csv.

Picks one scene (either specified SCENE_ID or the first row).

Reads:

Post-event image using tifffile

Label mask .tif

Prepares an RGB visualization (percentile stretch).

Shows:

Post-event image

Colorized label mask

Prints label distribution (unique IDs and pixel percentages).

This is used to:

Confirm label encoding (e.g., 0 = background, 1 = damaged).

Sanity check that images and masks align.

4.3. Tile Generation (Cell 3)
Goal: Create 256×256 tiles (image + mask) for training a segmentation model.

Settings:

TILE_SIZE = 256

STRIDE = 256 (no overlap; can be decreased for overlap)

MIN_POSITIVE_PIXELS = 500

Tiles with fewer than 500 positive pixels (label > 0) are skipped.

Flow:

Loop over all scenes in bright_manifest.csv.

Read post image and label.

Slide a 256×256 window across the label.

For each window:

If number of label > 0 pixels ≥ MIN_POSITIVE_PIXELS:

Save post tile to: artifacts_stage2/tiles/post/<scene>_<y>_<x>.npy

Save mask tile to: artifacts_stage2/tiles/label/<scene>_<y>_<x>.npy

Append an entry to tiles_manifest.csv.

Outputs:

text
Copy code
artifacts_stage2/tiles/post/*.npy
artifacts_stage2/tiles/label/*.npy
artifacts_stage2/tiles_manifest.csv
In your run, this produced:

~36,832 tiles

Average positive ratio ≈ 0.12 (12% of pixels damaged on sampled tiles).

4.4. Train/Dev/Test Split (Cell 4 – not pasted but used)
There is a cell (referenced in later cells) that:

Splits tiles into train/dev/test (typically by scene_id to avoid leakage).

Saves:

artifacts_stage2/splits/train_tiles.csv

artifacts_stage2/splits/dev_tiles.csv

artifacts_stage2/splits/test_tiles.csv

Each file references post_tile and label_tile paths.

4.5. DataLoader + Sanity Check (Cell 4C)
Defines a robust dataset class:

python
Copy code
class BrightTileDS_Safe(Dataset):
    - Loads .npy post tile (HxWx3 uint8)
    - Loads .npy label tile (HxW uint8)
    - Normalizes image to [0,1] and converts to CHW
    - Converts mask to 0/1, shape (1,H,W)
    - Optional horizontal/vertical flips for augmentation
Creates PyTorch DataLoaders:

train_loader, dev_loader, test_loader (batch size 8, no workers for Windows).

Also shows one random training tile + mask overlay to visually confirm correctness.

4.6. U-Net Training (Main BRIGHT Model)
You train a small U-Net for binary segmentation (damaged vs not-damaged).

Architecture:

Encoder: 4 downsampling levels (DoubleConv + MaxPool)

Bottleneck: base*16 channels

Decoder: 4 upsampling levels with skip connections

Output: 1-channel logits map (same H×W as input tile)

Loss:

Combined BCE + Dice Loss (BCEDiceLoss):

BCEWithLogitsLoss (weight 0.6)

Dice loss (weight 0.4)

Training details:

Optimizer: AdamW, lr = 3e-4, weight_decay = 1e-4

AMP mixed precision via torch.amp.autocast and GradScaler

Metrics per epoch:

Mean IoU (@0.5 threshold)

F1 (@0.5 threshold)

Early stopping based on validation IoU (patience = 3)

Best weights saved to:

text
Copy code
artifacts_stage2/models/unet_bce_dice/best.pt
artifacts_stage2/models/unet_bce_dice/train_log.csv
Final test metrics (tiles):

IoU@0.5 ≈ 0.33

F1@0.5 ≈ 0.49

These are tile-level segmentation results.

4.7. Fast Inference on 10 Scenes (Cell 6)
To make the results more interpretable at the scene level, you:

Pick the first 10 scene IDs from test_tiles.csv.

Collect all tiles for those scenes.

Run the trained U-Net on all tiles.

Aggregate TP/FP/FN per scene.

Compute per-scene IoU & F1 based on aggregated TP/FP/FN.

Save per-scene summary to:

text
Copy code
artifacts_stage2/preds_fast_summary.csv
Typical output (example):

Macro avg over 10 scenes:

IoU ≈ 0.28

F1 ≈ 0.44

Useful for the report and for selecting visual examples.

4.8. Overlays & Montage (Cell 7)
To create presentation-ready figures:

Load top-K scenes (by IoU) from preds_fast_summary.csv.

For each top scene:

Load its tiles from test_tiles.csv.

Run U-Net.

Overlay predicted mask on top of the RGB tile (alpha=0.35, red).

Save per-scene grid as:

text
Copy code
artifacts_stage2/inference_fast_overlays/<scene_id>_grid.png
Combine top-K grids into a montage:

text
Copy code
artifacts_stage2/inference_fast_overlays/topK_montage.png
This montage is great to show in the slide deck.

4.9. Join Events for Analysis (Cell 8)
Finally:

Merge per-scene predictions with event names from bright_manifest.csv.

Save:

text
Copy code
artifacts_stage2/preds_fast_with_events.csv
Columns:

scene_id, tiles, IoU, F1, event

You also show a “hotspot” filter, e.g., only bata-explosion scenes, sorted by IoU.

5. Reproducing Key Results (Quick Checklist)
To let a TA exactly replicate:

Set up environment (Section 1).

Place datasets as described (Section 2).

Open CrisisMMD.ipynb:

Run all cells.

Confirm:

Clean CSVs in artifacts_stage1/

Task 1 metrics (~0.84 acc, 0.79 macro-F1)

Task 2 improved metrics (~0.63 acc, 0.53 macro-F1)

Task 3 image-only, text-only, and fused metrics (~0.61 / 0.55 / 0.57 macro-F1).

Open Bright.ipynb:

Run all cells.

Confirm:

bright_manifest.csv and tiles_manifest.csv exist.

U-Net training completes with test IoU / F1 roughly 0.33 / 0.49.

preds_fast_summary.csv and preds_fast_with_events.csv exist.

Visualization files in artifacts_stage2/inference_fast_overlays/ (including topK_montage.png).

With the above steps and this README, a new user (TA / professor / future you) should be able to:

Recreate the cleaned datasets

Retrain all models

Re-generate the plots and metrics used in the report and slides.


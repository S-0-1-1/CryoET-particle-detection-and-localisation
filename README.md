# Cryo-ET particle picking with 3D heatmaps

Localization and classification of five protein complexes in cryo-electron tomograms, for the Kaggle competition [CZII CryoET Object Identification](https://www.kaggle.com/competitions/czii-cryo-et-object-identification). Single model, late submission.

![Kaggle results](assets/kaggle_results.png)

## Method

**Preprocessing.** Each tomogram is normalized to zero mean and unit variance. Training uses random 96³ crops from the four reconstructions of each tomogram (denoised, CTF-deconvolved, IsoNet-corrected, weighted back-projection), 70% of them placed around a particle, with random flips, 90° rotations and intensity jitter. Test tomograms only use the denoised reconstruction. TS_5_4 is held out for validation.

**Model.** A 3D residual U-Net (10.1M parameters) whose decoder stops at half resolution. It predicts one heatmap per class, with a Gaussian of width r/2 at each particle center (r is the particle radius), and a second head regresses the IsoNet-corrected volume. Particles are the local maxima of the heatmaps above a per-class threshold.

**Loss.** Penalty-reduced focal loss on the heatmaps, with class weights following the competition metric, plus 0.2 × L1 loss on the IsoNet head.

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/cryoet-particle-picking.git
cd cryoet-particle-picking
pip install -r requirements.txt
```

Download the [competition data](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/data) and set `DATA` to its folder.

## Usage

Prepare the training data:

```bash
python prepare.py --root $DATA --out data/prep
```

Train (the best checkpoint on TS_5_4 is saved to `runs/best.pt`):

```bash
python train.py --prep data/prep --out runs
```

Evaluate on TS_5_4 and save the weights with the tuned thresholds:

```bash
python evaluate.py --root $DATA --weights runs/best.pt --export weights/heatmap_unet.pt
```

Predict and write the submission file:

```bash
python predict.py --root $DATA --weights weights/heatmap_unet.pt --out submission.csv
```

## Pretrained weights

The weights behind the result above are in the [v1.0 release](https://github.com/YOUR-USERNAME/cryoet-particle-picking/releases/tag/v1.0), with the thresholds used for the submission:

```bash
wget -P weights https://github.com/YOUR-USERNAME/cryoet-particle-picking/releases/download/v1.0/heatmap_unet.pt
python predict.py --root $DATA --weights weights/heatmap_unet.pt --out submission.csv
```

## Data and license

Data from the CZII CryoET Object Identification competition, described in Peck et al., *A realistic phantom dataset for benchmarking cryo-ET data annotation*, Nature Methods, 2025. Code and weights under the MIT license.

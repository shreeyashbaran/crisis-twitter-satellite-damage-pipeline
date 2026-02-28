# Dataset Access (Not included in this repo)

This repository **does not** include the raw datasets (they are large and have their own licenses/terms).
Follow the official sources below and place the files in the expected folder structure.

---

## 1) CrisisMMD v2.0 (social media)

Official page: https://crisisnlp.qcri.org/crisismmd.html  
Official GitHub: https://github.com/CrisisMMD/CrisisMMD

### Expected folder (recommended)

Place the **v2.0 package** under the repo root:

```
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
```

> If your download uses an alternative layout (e.g., `all_json/`, `tsv/`, `image/`),
> you can still use this repo—just point `CRISISMMD_ROOT` to the folder that contains the
> *train/dev/test TSV split files* and the image directory. See the notebooks for details.

---

## 2) BRIGHT (satellite)

Paper: https://essd.copernicus.org/articles/17/6217/2025/  
Zenodo (dataset): https://zenodo.org/records/15385983  
(Dataset used for this project: Zenodo record 15385983; it includes the `post-event/` and `target/` folders.)

### Expected folder

Place BRIGHT under the repo root:

```
BRIGHT/
  post-event/
    <scene>_post_disaster.tif
    ...
  target/
    <scene>_building_damage.tif
    ...
  pre-event/    (optional; not required for this project)
```

---

## Storage note

BRIGHT (plus generated tiles/checkpoints) can easily exceed **30–40 GB**.
This repo ignores `artifacts_stage*/` by default.

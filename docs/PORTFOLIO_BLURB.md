# Portfolio blurb (copy/paste)

Built a two-stage, multimodal disaster-assessment pipeline:

1) Fine-tuned tweet-specialized transformers (BERTweet + Twitter-RoBERTa) on **CrisisMMD v2.0** for informativeness + humanitarian classification, and combined **text + image** damage-severity predictions via late fusion (best Macro-F1 ≈ 0.57).

2) Trained a lightweight **U-Net** on **BRIGHT** post-event satellite imagery, using **256×256 tiling** and BCE+Dice loss to segment damaged buildings (dev IoU 0.337; 10-scene test IoU 0.284).

Emphasis: reproducible preprocessing, class-imbalance handling, multimodal fusion, and interpretable qualitative overlays.

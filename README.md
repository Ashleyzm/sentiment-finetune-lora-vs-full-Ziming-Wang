# sentiment-finetune-lora-vs-full-Ziming-Wang
DistilBERT Sentiment: Full Fine-tuning vs LoRA

# DistilBERT Sentiment: Full Fine-tuning vs LoRA

Reproducible comparison on **SetFit/tweet_sentiment_extraction** (3 classes: **0=Negative, 1=Neutral, 2=Positive**) using **DistilBERT**.
We fix **seed=2002**, create a **stratified 10% validation split**, select the best model by **VALID Weighted-F1** (`metric_for_best_model="eval_f1"`, `load_best_model_at_end=True`), and **report final metrics once on TEST**.

## 1) Environment

```bash
pip install -r requirements.txt
```

**Versions used:**

```
transformers==4.44.2
datasets==2.19.1
peft==0.12.0
scikit-learn
matplotlib
statsmodels
emoji
pandas
torch 
```

## 2) Reproduce (one command)

```bash
bash run.sh
```

This will:

* download the HF dataset `SetFit/tweet_sentiment_extraction` and base model `distilbert-base-uncased`;
* train **Full Fine-tuning** (lr=2e-5, bs=16, epochs=2) and **LoRA** (r=8, alpha=16, dropout=0.1, lr=2e-4);
* evaluate on **TEST**;
* save all figures and CSVs to the working directory.

```bash
python main.py
```

## 3) What the code does

* **Data**: keep only `text`/`label`, tokenize with `max_length=128` (truncation & padding)
* **Split**: stratified 10% validation via `StratifiedShuffleSplit`
* **Training**: DistilBERT for 3-class classification; Full FT and LoRA (`target_modules=["q_lin","k_lin","v_lin"]`)
* **Metrics**: Accuracy, **Weighted-F1**, **Macro-F1**
* **Reports**: per-class classification report; **confusion matrix** images
* **Learning curves**: Train/Valid Loss, Valid Weighted-F1
* **Extras**: **Bootstrap 95% CI** & **McNemar** test; **overall** & **per-class** bar charts
* **Artifacts**: best checkpoints saved to `./best_full/` and `./best_lora/`

## 4) Outputs (after a successful run)

**Figures**

```
full_confusion_test.png
lora_confusion_test.png
full_loss_curve.png
full_f1_curve.png
lora_loss_curve.png
lora_f1_curve.png
overall_bar.png
perclass_precision.png
perclass_recall.png
perclass_f1.png
```

**CSV files**

```
test_predictions_full_lora.csv
full_errors.csv
lora_errors.csv
disagree_full_better.csv
disagree_lora_better.csv
```

**Models**

```
./best_full/
./best_lora/
```

## 5) Reproducibility notes

* Seed aligned for NumPy/Torch/CUDA: **2002**
* Validation: **10% stratified split** on the original training set
* Model selection: **VALID Weighted-F1** with best-checkpoint restore
* Final numbers reported **once on TEST**

## 6) Data & models (no large uploads)

We **do not** upload datasets or pretrained weights.
They are fetched automatically from Hugging Face:

```
Dataset: SetFit/tweet_sentiment_extraction
Base model: distilbert-base-uncased
```

## 7) Troubleshooting

* **Out of memory**: reduce batch size or `max_length`; FP16 is enabled by default.
* **No figures shown** in notebooks: check the files exist in the working directory and display with:

```python
from IPython.display import Image, display
display(Image('full_confusion_test.png'))
```

* **Private repo**: please add TAs as collaborators (Settings → Collaborators).

## 8) Minimal repo layout

```
.
├── README.md
├── requirements.txt
├── run.sh
├── main.py
└── (generated at runtime)  best_full/  best_lora/  *.png  *.csv
```


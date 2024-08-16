# VIMMCQA-PLMS

VIMMCQA-plms is a project for Vietnamese medical multiple-choice question answering using transformer models.

## Description

This script trains, validates, and tests a model for answering Vietnamese medical multiple-choice questions.

## Installation

1. Clone the repository:
   ```cmd
   git clone https://github.com/Ndphuong-17/VIMMCQA_PLMS.git
   cd VIMMCQA
   ```
2. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

## Running the Script

### Command:
```cmd
python main.py \
--train_file data_test/train1.csv \
--validation_file data_test/val1.csv \
--test_file data_test/test1.csv \
--old_wseg_corpus_file Corpus/164750_wseg_corpus.txt \
--output_dir output \
--model_name_or_path ndpphuong/medical_vietnamese_bi_encoder_finetune_simcse_part_2 \
--dimension 768 \
--task VIMMCQA \
--num_choices 4 \
--per_device_train_batch_size 256 \
--num_train_epochs 3 \
--test_index 0 \
--set_wseg True \
--test True \
--validation True \
--train True
```

### Parameters:
- `--train_file`: Path to the training CSV file.
- `--validation_file`: Path to the validation CSV file.
- `--test_file`: Path to the test CSV file.
- `--old_wseg_corpus_file`: Path to the word-segmented corpus file.
- `--output_dir`: Directory to save output data.
- `--model_name_or_path`: Pretrained model or model identifier from Hugging Face.
- `--dimension`: Dimension of the model embeddings (default: 768).
- `--task`: Task to perform (default: VIMMCQA).
- `--num_choices`: Number of choices in multiple-choice questions (default: 4).
- `--per_device_train_batch_size`: Batch size per device for training (default: 256).
- `--num_train_epochs`: Number of training epochs (default: 3).
- `--test_index`: Index of the test dataset (default: 0).
- `--set_wseg`: Whether to segment sentences before tokenization (default: True).
- `--test`: Whether to run testing (default: True).
- `--validation`: Whether to run validation (default: True).
- `--train`: Whether to run training (default: True).

---

### Notes:
- Ensure paths and model names are correctly specified based on your setup.
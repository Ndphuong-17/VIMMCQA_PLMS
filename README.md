# VIMMCQA-PLMS Model Training, Validation, and Testing

This script allows you to run training, validation, or testing for the VIMMCQA model, a multiple-choice question-answering model tailored for the Vietnamese medical context. The script is highly configurable with several arguments to fine-tune the model training process and manage different tasks.

## Usage

### Installation

1. Clone the repository:
   ```cmd
   git clone https://github.com/Ndphuong-17/VIMMCQA_PLMS.git
   cd VIMMCQA
   ```
2. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

### Command:

To run the script, use the following command structure:

```cmd
python main.py [options]
```

#### Parameters:

- `--train_file` (str):    Path to the training CSV file.  **Default:** `None`

- `--validation_file` (str):  Path to the validation CSV file.  **Default:** `None`

- `--test_file` (str):  Path to the test CSV file.  **Default:** `None`

- `--old_wseg_corpus_file` (str):  Path to the wseg documentary corpus txt file.  **Default:** `'Corpus/wseg_corpus.txt'`

- `--task` (str):  Specifies the task to be performed by the model.  **Default:** `'VIMMCQA'`  
  **Description:** The `VIMMCQA` task is a standard multiple-choice question-answering task without context retrieval. If you want to retrieve relevant context information, use the `full_VIMMCQA` task.

- `--output_dir` (str):  Directory for saving outputs and the trained model.  **Default:** `'output'`

- `--model_directory` (str):  Directory to load the model from, if resuming training or using a pre-trained model.  **Default:** `None`

- `--model_name_or_path` (str):  The name or path of the model to be used.  **Default:** `'ndpphuong/medical_vietnamese_bi_encoder_finetune_simcse_part_2'`

- `--dimension` (int):  The dimensionality of the model, depending on the `model_name_or_path`.  **Default:** `768`

- `--per_device_train_batch_size` (int):  The batch size for training.  **Default:** `256`

- `--num_train_epochs` (int):  Number of epochs to train the model.  **Default:** `3`

- `--test_index` (int):  Index of the test set to be used.  **Default:** `0`

- `--train` (bool):  Set to `True` to perform training.  **Default:** `False`

- `--validation` (bool):  Set to `True` to perform validation.  **Default:** `False`

- `--test` (bool):  Set to `True` to perform testing.  **Default:** `False`

---


## Examples

1. **Full features with none context retrieval**

   ```cmd
   python main.py \
   --train_file data_test/train1.csv \
   --validation_file data_test/val1.csv \
   --test_file data_test/test1.csv \
   --output_dir output \
   --task VIMMCQA \
   --model_name_or_path ndpphuong/medical_vietnamese_bi_encoder_finetune_simcse_part_2 \
   --dimension 768 \
   --per_device_train_batch_size 256 \
   --num_train_epochs 3 \
   --test True \
   --validation True \
   --train True
   ```

2. **Testing with a pre-trained model and none context retrieval**

   ```cmd
   python main.py \
   --test_file data_test/test1.csv \
   --output_dir output \
   --task VIMMCQA \
   --model_directory Result \
   --model_name_or_path ndpphuong/medical_vietnamese_bi_encoder_finetune_simcse_part_2 \
   --dimension 768 \
   --test True \
   ```

4. **Using full_VIMMCQA with context retrieval:**

   ```cmd
    python main.py --task full_VIMMCQA --train_file path/to/train.csv --train True
    ```

## Notes:
- Ensure paths and model names are correctly specified based on your setup.

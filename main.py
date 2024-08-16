import os
import torch
import argparse
import logging
import json
from transformers import TrainingArguments, set_seed, Trainer
from datasets import load_dataset
from src.Model import VIMMCQA, DataCollator, compute_metric
from sklearn.model_selection import train_test_split
import torch.nn as nn

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv file containing the training data.",
        required=True
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv file containing the validation data."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A csv file containing the test data."
    )
    parser.add_argument(
        "--old_wseg_corpus_file",
        type=str,
        default=None,
        help="A txt file containing the documentary corpus that has been word segmented, stored and has had corpus vector."
    )
    parser.add_argument(
        "--set_wseg",
        type=bool,
        default=True,
        help="Whether to segment sentences before tokenization or not."
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=True,
        help="Whether to test or not. Set to True to perform testing, False otherwise."
    )
    parser.add_argument(
        "--validation",
        type=bool,
        default=True,
        help="Whether to perform validation or not. Set to True to enable validation, False to disable it."
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=True,
        help="Whether to perform training or not. Set to True to enable training, False to disable it."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='',
        help="A directory to save output data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='ndpphuong/medical_vietnamese_bi_encoder_finetune_simcse_part_2',
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=768,
        help="The dimension of the model embeddings."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="VIMMCQA",
        help="The task to perform."
    )
    parser.add_argument(
        "--num_choices",
        type=int,
        default=4,
        help="Number of options in multiple-choice question answering."
    )
    parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=256,
        help="Batch size per device for training."
    )
    parser.add_argument(
        '--num_train_epochs',
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        '--test_index',
        type=int,
        default=0,
        help="Index of the test dataset to use."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    print("model_name_or_path: ", args.model_name_or_path)
    print("dimension: ", args.dimension)
    print("per_device_train_batch_size: ", args.per_device_train_batch_size)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_pin_memory=False,
        report_to=['tensorboard'],
        remove_unused_columns=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
    )

    # Set seed before initializing model
    set_seed(training_args.seed)

    # SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
    os.environ['HF_HOME'] = os.path.join(".", "cache")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    if args.test_index == 0:
        dataset = load_dataset('csv', data_files={
            'train': args.train_file,
            'test': args.test_file if args.test_file else args.train_file,
            'val': args.validation_file if args.validation_file else args.train_file
        })
    else:
        raise KeyError(f"Did not set up for testing {args.test_index} samples.")

    # Load segmented corpus
    with open(args.old_wseg_corpus_file, 'r', encoding='utf-8') as _file:
        datas = _file.read()
    wseg_datas = datas.split('\n')
    print(len(wseg_datas))
    non_wseg_datas = datas.replace('_', ' ').split('\n')

    print("Initializing corpus wseg_datas completely.")
    print("Initializing corpus non_wseg_datas completely.")

    # Initialize data collator
    data_collator = DataCollator(
        model_args=args,
        corpus=wseg_datas
    )
    print("Initializing dataCollator completely.")
    print(data_collator)

    # Initialize model
    model = VIMMCQA(model_args=args)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    print("Initializing model completely.")

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        data_collator=data_collator,
    )

    # Show number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")
    print(f"Total parameters: {total_params}")

    # Training
    print('Training...')
    if args.train:
        logger.info("*** Training ***")
        train_result = trainer.train()
        print(train_result)
        print("Training process finished")

    # Evaluation
    print('Evaluation...')
    if args.validation and args.validation_file is not None:
        logger.info("*** Evaluation ***")
        eval_metrics = trainer.evaluate()
        print(eval_metrics)
        print("Evaluation process finished")

    # Testing
    print('Testing...')
    if args.test and args.test_file is not None:
        logger.info("*** Testing ***")
        predictions = trainer.predict(dataset['test'], metric_key_prefix="predict").predictions
        print("Testing process finished")

        # Test results
        print("--- Test Results ---")
        predictions_tensor = torch.Tensor(predictions[1])
        labels_tensor = torch.tensor([eval(s) for s in dataset['test']['label']], dtype=torch.float)
        metrics = compute_metric(predictions_tensor, labels_tensor)
        print(metrics)

        # Convert tensors to lists
        predictions_list = predictions_tensor.tolist()
        labels_list = labels_tensor.tolist()

        # Prepare data with id field
        data = [
            {
                "id": i,
                "pred": pred,
                "label": label
            }
            for i, (pred, label) in enumerate(zip(predictions_list, labels_list))
        ]

        # Save data to JSON file
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(data, f, indent=4)
        print("Test data saved to results.json")
        
    # Save the model, tokenizer, and training arguments
    model.save_pretrained(args.output_dir)
    trainer.save_state()

if __name__ == "__main__":
    main()

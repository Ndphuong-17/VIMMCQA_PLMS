import os
import torch
import argparse
import logging
import json
from transformers import TrainingArguments, set_seed, Trainer
from datasets import load_dataset
from src.Model import VIMMCQA, DataCollator, compute_metrics, compute_metric
import torch.nn as nn
import pandas as pd
import math


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run VIMMCQA model training, validation, or testing")

    parser.add_argument('--train_file', type=str, default='data_test/train1.csv', help="Path to the training csv file")
    parser.add_argument('--validation_file', type=str, default='data_test/val1.csv', help="Path to the validation csv file")
    parser.add_argument('--test_file', type=str, default='data_test/test1.csv', help="Path to the test csv file")
    parser.add_argument('--old_wseg_corpus_file', type=str, default='Corpus/wseg_corpus.txt', help="Path to the wseg documentary corpus txt file")

    parser.add_argument('--task', type=str, default='VIMMCQA', 
    help="Default VIMMCQA Just MCQA task not supported retrieving. If you want to retrieve relevant context information, use full_VIMMCQA"
    )

    parser.add_argument('--output_dir', type=str, default='output', help="Directory for saving outputs and model")

    parser.add_argument('--model_directory', type=str, default=None, help="Directory to load the model")
    parser.add_argument('--model_name_or_path', type=str, default='ndpphuong/medical_vietnamese_bi_encoder_finetune_simcse_part_2', help="Model name or path")

    parser.add_argument('--dimension', type=int, default=768, help="Dimension of the model, based on the model_name_or_path")
    parser.add_argument('--per_device_train_batch_size', type=int, default=256, help="Batch size for training")
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for labeling")
    parser.add_argument('--max_length', type=int, default=512, help="Label smoothing")

    parser.add_argument('--num_choices', type=int, default=4, help="Number of choices for the task")
    parser.add_argument('--test_index', type=int, default=0, help="Index of the test set")
    parser.add_argument('--set_wseg', action='store_true', help="Flag to set wseg")

    parser.add_argument('--test', default = False, help="Set to True to perform testing")
    parser.add_argument('--validation',default = False, help="Set to True to perform validation")
    parser.add_argument('--train',default = False, help="Set to True to perform training")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create the directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
        per_device_eval_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="epoch",
        logging_dir= os.path.join(args.output_dir, "logs"),
        fp16=False,  # Enable mixed precision
        eval_accumulation_steps=10,  # Accumulate gradients over 10 steps during evaluation

    )

    # Set seed before initializing model
    set_seed(training_args.seed)

    # SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
    os.environ['HF_HOME'] = os.path.join(".", "cache")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    if args.train_file and args.test_index == 0:
        dataset = load_dataset('csv', data_files={
            'train': args.train_file,
            'test': args.test_file if args.test_file else args.train_file,
            'val': args.validation_file if args.validation_file else args.train_file
        })
    elif args.train_file and args.test_index > 0:
        pd.read_csv(args.train_file)[: args.test_index]\
            .to_csv(os.path.join(args.output_dir, f'train_{args.test_index}.json'))
        pd.read_csv(args.test_file if args.test_file else args.train_file)[: args.test_index]\
            .to_csv(os.path.join(args.output_dir, f'test_{args.test_index}.json'))
        pd.read_csv(args.validation_file if args.validation_file else args.train_file)[: args.test_index]\
            .to_csv(os.path.join(args.output_dir, f'val_{args.test_index}.json'))
        dataset = load_dataset('csv', data_files={
            'train': os.path.join(args.output_dir, f'train_{args.test_index}.json'),
            'test':  os.path.join(args.output_dir, f'test_{args.test_index}.json'),
            'val':   os.path.join(args.output_dir, f'val_{args.test_index}.json')
        })

    elif args.test_file and args.model_directory and args.test_index == 0:
        dataset = load_dataset('csv', data_files={
            'train': args.test_file,
            'test': args.test_file,
            'val': args.test_file
        })
    elif args.test_file and args.model_directory and args.test_index > 0:
        pd.read_csv(args.test_file if args.test_file else args.train_file)[: args.test_index]\
            .to_csv(os.path.join(args.output_dir, f'test_{args.test_index}.json'))
        
        dataset = load_dataset('csv', data_files={
            'train': os.path.join(args.output_dir, f'test_{args.test_index}.json'),
            'test':  os.path.join(args.output_dir, f'test_{args.test_index}.json'),
            'val':   os.path.join(args.output_dir, f'test_{args.test_index}.json')
        })
    elif args.test_index == 0:
        raise ValueError("Either train_file or (test_file, model_directory) must be provided.")
    else:
        raise KeyError(f"Did not set up for testing {args.test_index} (< 0) samples.")


    if args.task == 'VIMMCQA':
        wseg_datas = None

    elif args.task == 'full_VIMMCQA':
        # Load segmented corpus
        with open(args.old_wseg_corpus_file, 'r', encoding='utf-8') as _file:
            datas = _file.read()
        wseg_datas = datas.split('\n')
        print(len(wseg_datas))
        print("Initializing corpus wseg_datas completely.")

    if args.model_directory:
        model = VIMMCQA.from_pretrained(args.model_directory, model_args=args)
        print("Loading model completely.")
    else:
        # Initialize model
        model = VIMMCQA(model_args=args)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        print("Initializing model completely.")


    # Initialize data collator
    data_collator = DataCollator(
        model_args=args,
        corpus=wseg_datas
    )
    print("Initializing dataCollator completely.")
    print(data_collator)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # Pass the compute_metrics function
    )
    # Show number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")
    print(f"Total parameters: {total_params}")

    # Training
    if args.train:
        print('Training...')    
        logger.info("*** Training ***")
        train_result = trainer.train()
        print(train_result)
        print("Training process finished")

        # Save the model, tokenizer, and training arguments
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        model.save_pretrained(args.output_dir)
        try:
            trainer.save_state()
        except:
            print("Failed to save training state")
            pass

    # Evaluation
    if args.validation and args.validation_file is not None:
        print('Evaluation...')
        logger.info("*** Evaluation ***")
        eval_metrics = trainer.evaluate()
        print(eval_metrics)
        print("Evaluation process finished")

    # Testing
    if args.test and args.test_file is not None:
        print('Testing...')
        logger.info("*** Testing ***")
        batch_size = 1 # args.per_device_train_batch_size // 2
        num_batches = math.ceil(len(dataset['test']) / batch_size)
        all_predictions = []
        all_labels = []
        
        for i in range(num_batches):
            print(f"Processing batch {i+1}/{num_batches}")
            batch = dataset['test'].select(range(i * batch_size, min((i + 1) * batch_size, len(dataset['test']))))
            
            torch.cuda.synchronize()

            # Add a try-except block to catch and log errors during testing
            try:
                predictions = trainer.predict(batch, metric_key_prefix="predict").predictions
                all_predictions.extend(predictions)
                all_labels.extend([eval(s) for s in batch['label']])

            except:
                print(batch)
                pass

            torch.cuda.synchronize()
            # Clear GPU cache to free up memory
            torch.cuda.empty_cache()

        print("Testing process finished")


        # Test results
        print("--- Test Results ---")
        predictions_tensor = torch.Tensor(all_predictions)
        labels_tensor = torch.Tensor(all_labels)
        # labels_tensor = torch.tensor([eval(s) for s in dataset['test']['label']], dtype=torch.float)
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
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.save_pretrained(args.output_dir)
    try:
        trainer.save_state()
    except:
        print("Failed to save training state")
        pass


if __name__ == "__main__":
    main()

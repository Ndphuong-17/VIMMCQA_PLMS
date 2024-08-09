import subprocess
import sys

def run_main_script():
    # Build the command to run main.py
    command = [sys.executable, "main.py"]

    # Add additional command line arguments if needed
    command.extend([
        "--train_file", r"D:\Project\VIMMCQA\Model\runs\checkpoint\train1.csv",
        "--validation_file", r"D:\Project\VIMMCQA\Model\runs\checkpoint\val1.csv",
        "--test_file", r"D:\Project\VIMMCQA\Model\runs\checkpoint\train1.csv",
        "--old_wseg_corpus_file", r"D:\Project\VIMMCQA\Data\164750_wseg_corpus.txt",
        "--output_dir", r"D:\Project\VIMMCQA\Model\runs1",
        "--model_name_or_path", "ndpphuong/medical_vietnamese_bi_encoder_finetune_simcse_part_2",
        "--dimension", "768",
        "--task", "VIMMCQA",
        "--num_choices", "4",
        "--per_device_train_batch_size", "256",
        "--num_train_epochs", "3",
        "--test_index", "0",
        # Set additional boolean flags
        "--set_wseg", "True",
        "--test", "True",
        "--validation", "True",
        "--train", "True"
    ])

    # Run the main.py script
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output and error (if any)
    print("Output:\n", result.stdout)
    if result.stderr:
        print("Error:\n", result.stderr)

if __name__ == "__main__":
    run_main_script()

"""
Fine-tunes a pre-trained Transformer model (e.g., GPT-2) on a custom dataset
of Haiku poems to create a Haiku-generating chatbot.

This script covers:
- Loading a dataset of haikus.
- Loading a pre-trained model and tokenizer from Hugging Face Transformers.
- Setting up and running the fine-tuning process using the Trainer API.
- Running an interactive inference loop to chat with the fine-tuned model.

Corresponds to the final project in Module 7 of the curriculum.
"""

# --- Imports ---
# Standard PyTorch
import torch

# Hugging Face Transformers essentials
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TextDataset,  # A simple way to load text data
)
import os
import random  # Maybe useful for prompts or seeding

# --- Configuration & Parameters ---
# Model choice (e.g., 'gpt2', 'distilgpt2')
PRETRAINED_MODEL_NAME = "distilgpt2"  # Smaller, faster to train
# Data path
DATASET_PATH = "haiku_dataset.txt"
# Fine-tuning output
OUTPUT_DIR = "./haiku_bot_finetuned"
# Training parameters
NUM_TRAIN_EPOCHS = 3  # Adjust as needed
PER_DEVICE_TRAIN_BATCH_SIZE = 4  # Adjust based on GPU memory
SAVE_STEPS = 500  # Save checkpoints periodically


# --- 1. Data Preparation ---
def prepare_dataset(tokenizer, file_path):
    """Loads and tokenizes the text dataset for language modeling.

    Uses `TextDataset` for simplicity, which loads the entire file.
    For very large datasets, consider using the `datasets` library for streaming/memory mapping.

    Args:
        tokenizer: The Hugging Face tokenizer instance.
        file_path (str): Path to the text file containing the training data (e.g., haikus).

    Returns:
        datasets.Dataset: The prepared dataset object.
    """
    print(f"Loading dataset from {file_path}...")
    # TextDataset is simple but loads everything into memory
    # For larger datasets, consider `load_dataset` from `datasets` library
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128,  # Max sequence length for training chunks
    )
    print(f"Dataset loaded with {len(dataset)} examples.")
    return dataset


# --- 2. Model Loading ---
def load_model_and_tokenizer(model_name):
    """Loads the specified pre-trained causal language model and its tokenizer.

    Handles setting the padding token for models like GPT-2.

    Args:
        model_name (str): The name of the pre-trained model (e.g., 'gpt2', 'distilgpt2').

    Returns:
        tuple: A tuple containing (model, tokenizer).
    """
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad token if not present (GPT-2 typically doesn't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Resize embeddings if pad token was added
    model.resize_token_embeddings(len(tokenizer))
    print("Model and tokenizer loaded.")
    return model, tokenizer


# --- 3. Fine-Tuning Setup ---
def setup_training(tokenizer, dataset):
    """Configures the DataCollator and TrainingArguments for fine-tuning.

    Args:
        tokenizer: The Hugging Face tokenizer instance.
        dataset (datasets.Dataset): The training dataset.

    Returns:
        tuple: A tuple containing (data_collator, training_args).
    """
    # Data Collator: Handles batching and padding for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # MLM is for BERT-like models
    )

    # Training Arguments: Configure the training process
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        save_steps=SAVE_STEPS,
        save_total_limit=2,  # Keep only the latest 2 checkpoints
        logging_dir="./logs",  # Directory for TensorBoard logs (view with `tensorboard --logdir=./logs`)
        logging_steps=100,
        # Add fp16=True if you have a capable GPU and want faster training
        # fp16=torch.cuda.is_available(),
    )
    print("Training arguments set.")
    return data_collator, training_args


# --- 4. Training Execution ---
def run_training(model, tokenizer, dataset, data_collator, training_args):
    """Initializes the Hugging Face Trainer and runs the fine-tuning process.

    Saves the final model and tokenizer to the specified output directory.

    Args:
        model: The model to be fine-tuned.
        tokenizer: The tokenizer associated with the model.
        dataset (datasets.Dataset): The training dataset.
        data_collator: The data collator for batching.
        training_args (TrainingArguments): The training configuration.
    """
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        # eval_dataset=... # Optionally add an evaluation dataset
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning finished.")

    # Save the final model and tokenizer
    print(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model and tokenizer saved.")


# --- 5. Inference/Chat Loop ---
def run_inference(model_path):
    """Loads a fine-tuned model and runs an interactive chat loop.

    Prompts the user for input, generates a response using the model,
    formats it slightly, and prints the output.

    Args:
        model_path (str): Path to the directory containing the fine-tuned model and tokenizer.
    """
    print(f"\n--- Loading Fine-Tuned Haiku Bot from {model_path} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")

    print("\nEnter your prompt (or type 'quit' to exit).")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        # Basic prompt formatting (can be improved)
        # It might be better to include examples in the prompt if needed
        prompt = f"User: {user_input}\nAssistant (in Haiku):"

        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate response
        print("Haiku Bot: ... *composing verse* ...")
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=len(input_ids[0]) + 50,  # Max length for the response part
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,  # Stop generation at EOS
                do_sample=True,  # Use sampling for more creative output
                top_k=50,  # Consider top K tokens
                top_p=0.95,  # Use nucleus sampling
                temperature=0.7,  # Control randomness (lower = more focused)
            )

        # Decode and print the generated text
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        # Extract only the assistant's part (simple split)
        assistant_response = generated_text.split("Assistant (in Haiku):")[-1].strip()

        # Basic formatting (ensure 3 lines if possible)
        # Note: This just takes the first 3 lines found after splitting by newline.
        # It does NOT guarantee 5-7-5 syllable structure or perfect haiku form.
        # Evaluating true haiku quality would require more sophisticated analysis.
        lines = assistant_response.split("\n")
        formatted_response = "\n".join(lines[:3])  # Take at most 3 lines

        print(f"Haiku Bot:\n{formatted_response}")
        print("---")

    print("Exiting Haiku Bot. Farewell!")


# --- Main Execution Logic ---
if __name__ == "__main__":
    """Main execution block.

    Handles checking for the dataset, running either the fine-tuning phase
    or the inference phase based on commented-out sections and the presence
    of a previously fine-tuned model.
    """
    # --- Phase 1: Fine-tuning (Run this part first) ---
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset file not found at {DATASET_PATH}")
        print(
            "Please create this file and populate it with Haiku poems, one per line, separated by newlines."
        )
        # Example haiku_dataset.txt content:
        # Old pond, still frog leaps -
        # sound echoes in the water.
        #
        # First autumn morning
        # the mirror I stare into
        # shows my father's face.
        exit(1)

    # model, tokenizer = load_model_and_tokenizer(PRETRAINED_MODEL_NAME)
    # dataset = prepare_dataset(tokenizer, DATASET_PATH)
    # data_collator, training_args = setup_training(tokenizer, dataset)
    # run_training(model, tokenizer, dataset, data_collator, training_args)
    # print("\n--- Fine-tuning complete. Model saved in ", OUTPUT_DIR, "---")
    # print("Comment out the training lines and uncomment the inference line below to chat.")

    # --- Phase 2: Inference (Run this after fine-tuning) ---
    # Ensure the fine-tuned model exists before running inference
    if os.path.exists(OUTPUT_DIR) and os.path.exists(
        os.path.join(OUTPUT_DIR, "pytorch_model.bin")
    ):
        run_inference(model_path=OUTPUT_DIR)
    else:
        print(f"\nFine-tuned model not found in {OUTPUT_DIR}.")
        print(
            "Please run the fine-tuning phase first (uncomment training lines above)."
        )

    # To run only inference after training, comment out the training lines (approx lines 119-122)
    # and make sure the inference line (approx line 128) is uncommented.

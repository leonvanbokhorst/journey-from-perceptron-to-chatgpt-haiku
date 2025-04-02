### Final Project: The Haiku Bot – Distilling Wisdom into Verse (Fine-Tuning Path)

This project brings our journey full circle. We will leverage a pre-trained Transformer model (like GPT-2 or DistilGPT2) and fine-tune it on a collection of haiku poems, teaching it to respond to user prompts in haiku form.

**A Guide to Crafting the Haiku Bot via Fine-Tuning**

1.  **Gather the Verses (Prepare `haiku_dataset.txt`):**

    - **Action:** Create a plain text file named `haiku_dataset.txt` in the `module_07_final_project` directory.
    - **Content:** Populate this file with haiku poems. Each haiku should ideally occupy three lines, and different haiku should be separated by at least one blank line.
    - **Example Format:**

      ```
      Old pond, still frog leaps -
      sound echoes in the water.

      First autumn morning
      the mirror I stare into
      shows my father's face.

      O snail
      Climb Mount Fuji,
      But slowly, slowly!
      ```

    - **Source:** Find public domain haiku online (Bashō, Issa, Buson, Shiki are good starting points). Aim for at least a few dozen, but more (hundreds) is better for fine-tuning.
    - **Importance:** The quality and format of this dataset significantly impact the fine-tuning result.

2.  **Select the Brush & Ink (Imports and Configuration):**

    - The script `haiku_bot_finetune.py` imports necessary libraries from `torch` and Hugging Face `transformers`.
    - Key configurations are set near the top:
      - `PRETRAINED_MODEL_NAME`: Choose a base model (e.g., `"distilgpt2"` for faster training, `"gpt2"` for potentially better results).
      - `DATASET_PATH`: Should point to your `haiku_dataset.txt`.
      - `OUTPUT_DIR`: Where the fine-tuned model will be saved.
      - Training parameters (`NUM_TRAIN_EPOCHS`, `PER_DEVICE_TRAIN_BATCH_SIZE`, etc.) can be adjusted based on your hardware and time.

    ```python
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, ...
    )
    import os
    import torch

    PRETRAINED_MODEL_NAME = "distilgpt2"
    DATASET_PATH = "haiku_dataset.txt"
    OUTPUT_DIR = "./haiku_bot_finetuned"
    # ... other params
    ```

3.  **Prepare the Inkstone (Data Loading & Tokenization):**

    - The `load_model_and_tokenizer` function fetches the specified pre-trained model and its corresponding tokenizer.
    - It handles setting a `pad_token` if the base model (like GPT-2) doesn't have one, usually setting it to the `eos_token` (end-of-sequence).
    - The `prepare_dataset` function uses Hugging Face's `TextDataset` to load your `haiku_dataset.txt` and tokenize it according to the model's tokenizer. It chunks the text into blocks (`block_size`).

    ```python
    # Inside main execution block:
    # model, tokenizer = load_model_and_tokenizer(PRETRAINED_MODEL_NAME)
    # dataset = prepare_dataset(tokenizer, DATASET_PATH)
    ```

4.  **Arrange the Tools (Training Setup):**

    - The `setup_training` function prepares:
      - `DataCollatorForLanguageModeling`: This handles creating batches from your dataset, including necessary padding for language modeling.
      - `TrainingArguments`: This object holds all the configuration for the `Trainer` (output directory, epochs, batch size, saving frequency, logging, etc.).

    ```python
    # Inside main execution block:
    # data_collator, training_args = setup_training(tokenizer, dataset)
    ```

5.  **Teach the Strokes (Fine-Tuning Execution):**

    - The `run_training` function initializes the Hugging Face `Trainer` with the model, training arguments, data collator, and dataset.
    - `trainer.train()` starts the fine-tuning process. This will take time depending on your dataset size, model size, and hardware (GPU highly recommended).
    - After training, `trainer.save_model()` saves the fine-tuned model weights and configuration to the specified `OUTPUT_DIR`.
    - The tokenizer configuration is also saved alongside the model.

    ```python
    # Inside main execution block (ensure these lines are uncommented for Phase 1):
    # run_training(model, tokenizer, dataset, data_collator, training_args)
    ```

6.  **Awaken the Poet (Inference/Chat Loop):**
    - **Important:** After fine-tuning is complete, **comment out** the lines that load the base model and run the training (`load_model_and_tokenizer`, `prepare_dataset`, `setup_training`, `run_training`).
    - **Uncomment** the section that calls `run_inference(model_path=OUTPUT_DIR)`.
    - The `run_inference` function:
      - Loads your _fine-tuned_ model and tokenizer from the `OUTPUT_DIR`.
      - Moves the model to the appropriate device (GPU or CPU).
      - Enters an interactive loop prompting the user for input.
      - Formats the user input into a simple prompt structure (e.g., `User: ...\nAssistant (in Haiku):`).
      - Uses `model.generate()` to produce a response based on the prompt. Generation parameters like `max_length`, `do_sample`, `top_k`, `top_p`, and `temperature` control the output's length, creativity, and randomness.
      - Decodes the generated token IDs back into text.
      - Attempts basic formatting to extract the assistant's response and limit it to roughly three lines.
      - Prints the Haiku Bot's response.
    ```python
    # Inside main execution block (ensure training lines are commented out for Phase 2):
    # if os.path.exists(OUTPUT_DIR) and ...:
    #      run_inference(model_path=OUTPUT_DIR)
    ```

**Running the Project:**

1.  **Prepare Data:** Create and populate `haiku_dataset.txt`.
2.  **Install Libraries:** Make sure you have `torch` and `transformers` installed (`pip install torch transformers`). You might also need `datasets` if using larger files later.
3.  **Run Phase 1 (Fine-Tuning):** Execute `python haiku_bot_finetune.py`. Ensure the training lines in the `if __name__ == "__main__":` block are active. Monitor the training progress. This might take minutes to hours.
4.  **Run Phase 2 (Inference):** Once training is complete and the model is saved in `OUTPUT_DIR`, **edit** `haiku_bot_finetune.py`. Comment out the lines related to training (as indicated in the script comments) and make sure the `run_inference` call is active. Execute `python haiku_bot_finetune.py` again. You should now be able to interact with your Haiku Bot.

**Reflection & Evaluation:**
Interact with your bot! Ask it questions, give it prompts. Evaluate the output:

- Does it respond in a haiku-like format (3 lines)?
- Is the tone appropriate?
- Does it capture the essence of haiku (nature, brevity, perhaps a 'kireji' or cut)?
- How well does it adhere to the 5-7-5 syllable structure (this is often approximate)?
- Is the response relevant to your prompt?

This project demonstrates the power of fine-tuning large pre-trained models to adopt specific styles and personas, bridging the gap between general language understanding and specialized creative tasks.

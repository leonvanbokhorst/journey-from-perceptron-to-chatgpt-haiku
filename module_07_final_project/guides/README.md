# Module 7 Guide: Final Project – Building a Haiku Chatbot

This guide provides context for the final project code in Module 7, which involves fine-tuning a Transformer model to create a Haiku-generating chatbot. For the full discussion on Large Language Models (LLMs), ChatGPT, RLHF, ethical considerations, and project rationale, please refer to **Module 7** in the main `../../curriculum.md` file.

## Objectives Recap

- Explore how Transformers enable Large Language Models (LLMs) like GPT.
- Understand the basics of fine-tuning a pre-trained model for a specific style/task.
- Use the Hugging Face `transformers` library to load models, tokenizers, and manage training.
- Build a functional, albeit simplified, chatbot that responds in a Haiku-like style.
- Reflect on the capabilities and limitations of modern generative AI.

## Code Example: `haiku_bot_finetune.py`

The script `../haiku_bot_finetune.py` implements the fine-tuning approach (Approach 1 described in the curriculum) to create the Haiku Bot.

**Key Components:**

1.  **Configuration:** Sets parameters like the base model (`PRETRAINED_MODEL_NAME`, defaulting to `distilgpt2`), dataset path (`DATASET_PATH`), output directory (`OUTPUT_DIR`), and basic training hyperparameters.
2.  **`prepare_dataset` Function:** Loads text data (expected to be haikus, one per line or clearly separated) from the specified file path using `TextDataset`.
3.  **`load_model_and_tokenizer` Function:** Loads the specified pre-trained model (`AutoModelForCausalLM`) and its corresponding tokenizer (`AutoTokenizer`) from Hugging Face. Handles adding a padding token if necessary.
4.  **`setup_training` Function:** Configures the necessary components for the `Trainer` API:
    - `DataCollatorForLanguageModeling`: Prepares batches suitable for causal language modeling (predicting the next token).
    - `TrainingArguments`: Sets various training options like output directory, epochs, batch size, save steps, logging, etc.
5.  **`run_training` Function:**
    - Initializes the `Trainer` object with the model, training args, dataset, collator, and tokenizer.
    - Calls `trainer.train()` to start the fine-tuning process.
    - Saves the final fine-tuned model and tokenizer to the specified `OUTPUT_DIR`.
6.  **`run_inference` Function:**
    - Loads the _fine-tuned_ model and tokenizer from the `OUTPUT_DIR`.
    - Enters an interactive loop:
      - Takes user input.
      - Formats a simple prompt (e.g., `"User: {user_input}\nAssistant (in Haiku):"`).
      - Encodes the prompt using the tokenizer.
      - Uses `model.generate()` to produce a text sequence based on the prompt. Generation parameters (`max_length`, `num_return_sequences`, `pad_token_id`, sampling parameters like `top_k`, `top_p`, `temperature`) control the output style and length.
      - Decodes the generated sequence back into text.
      - Extracts and prints the assistant's response part.
      - Includes basic formatting to limit output to roughly 3 lines (note: does not enforce syllable counts).
7.  **`if __name__ == \"__main__\":` Block:**
    - Manages the workflow: Checks if the required `haiku_dataset.txt` exists. Provides commented-out sections to run either the fine-tuning phase (Phase 1) or the inference phase (Phase 2), guiding the user to run training first and then switch to inference using the saved model.

**Running the Project:**

1.  **Prepare Data:** Create a file named `haiku_dataset.txt` in the main project directory (or adjust `DATASET_PATH` in the script). Populate this file with haiku poems, ideally one haiku per line, with lines separated by newlines. Ensure good quality and variety.
2.  **Fine-Tune:** Uncomment the training lines (approx. 177-180) in the `if __name__ == "__main__":` block and comment out the inference section (approx. 184-190). Run the script:
    ```bash
    cd module_07_final_project
    python haiku_bot_finetune.py
    ```
    This will take some time depending on your hardware (GPU highly recommended) and dataset size. It will save the fine-tuned model in the `OUTPUT_DIR` (default: `./haiku_bot_finetuned`).
3.  **Chat:** Once fine-tuning is complete, comment out the training lines again and uncomment the inference section. Run the script:
    ```bash
    cd module_07_final_project
    python haiku_bot_finetune.py
    ```
    The script will load your fine-tuned model, and you can interact with it by typing prompts.

**Expected Outcome:**
The fine-tuned bot should respond to your prompts with text that attempts to mimic the style and content of the haikus it was trained on. Quality will depend heavily on the dataset size/quality and training parameters. It likely won't perfectly adhere to 5-7-5 syllables without more advanced techniques, but it should capture the _essence_ of haiku far better than the base pre-trained model.

## Further Exploration (Beyond the Scope of Provided Code)

Refer back to **Module 7** in `../../curriculum.md` for discussions on:

- Alternative approaches (prompt engineering, rule-based post-processing).
- Using larger models (requires more compute/APIs).
- Implementing syllable counting or stricter format enforcement.
- Ethical considerations for LLMs.

This final project integrates the Transformer concepts into a practical, creative application, demonstrating the power of fine-tuning pre-trained models.

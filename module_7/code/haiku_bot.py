"""
Haiku Bot - Module 7
-------------------
A simple interface for a chatbot that responds with haiku.
This serves as a starting point for the final project.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import numpy as np
import random
import re
from typing import List, Dict, Any, Optional

# Import syllable tools
from syllable_counter import count_syllables_in_text, is_valid_haiku, format_as_haiku


class HaikuBot:
    """
    A chatbot that responds to user input with haiku poems.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the haiku bot.

        Args:
            model_path: Path to a fine-tuned GPT-2 model, or None to use the default model
        """
        # Load model and tokenizer
        if model_path:
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        else:
            # Use the default GPT-2 model (you would replace this with your fine-tuned model)
            print(
                "Loading default GPT-2 model. For better results, use a fine-tuned model."
            )
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Set the device (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Add special tokens if needed
        special_tokens = {
            "pad_token": "<PAD>",
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Conversation history (for context)
        self.conversation_history = []

        # Seasonal themes for haiku
        self.seasons = {
            "spring": [
                "blossom",
                "flower",
                "green",
                "birth",
                "rain",
                "renewal",
                "growth",
            ],
            "summer": ["sun", "heat", "beach", "ocean", "bright", "warm", "light"],
            "autumn": ["leaves", "fall", "orange", "harvest", "wind", "cool", "mist"],
            "winter": ["snow", "cold", "ice", "frost", "barren", "sleep", "silence"],
        }

        # Track the current season theme
        self.current_season = random.choice(list(self.seasons.keys()))

    def _prepare_input(self, user_input: str) -> str:
        """
        Prepare the user input for the model by formatting it with prompt.

        Args:
            user_input: The user's message

        Returns:
            Formatted input for the model
        """
        # Store conversation for context
        self.conversation_history.append(user_input)

        # Keep only the last few exchanges to avoid context getting too long
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]

        # Extract keywords from user input to guide the haiku
        keywords = self._extract_keywords(user_input)

        # Choose theme words based on current season
        season_words = self.seasons[self.current_season]
        theme_word = random.choice(season_words)

        # Create a prompt that includes some context and guidance
        prompt = f"Write a haiku about {', '.join(keywords)} with the theme of {theme_word}.\n\nHaiku:"

        return prompt

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract key words from the user input to guide the haiku generation.

        Args:
            text: The user input text

        Returns:
            List of keywords
        """
        # Remove punctuation and convert to lowercase
        text = re.sub(r"[^\w\s]", "", text.lower())

        # Split into words
        words = text.split()

        # Remove common stop words (you could use NLTK's stopwords for a more complete list)
        stop_words = {
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "to",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
        }

        # Filter out stop words
        keywords = [word for word in words if word not in stop_words and len(word) > 3]

        # If no meaningful keywords, use a default
        if not keywords:
            return ["nature"]

        # Return a limited number of keywords (to avoid overwhelming the model)
        return keywords[:3]

    def _generate_haiku(self, prompt: str) -> str:
        """
        Generate a haiku using the GPT-2 model.

        Args:
            prompt: The prepared input prompt

        Returns:
            Generated haiku text
        """
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Set generation parameters
        max_length = input_ids.shape[1] + 50  # Allow space for a complete haiku

        # Generate text
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.9,  # Higher temperature for more creativity
                top_k=50,  # Sample from top-k probable tokens
                top_p=0.95,  # Nucleus sampling
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract just the haiku part (after the prompt)
        if "Haiku:" in generated_text:
            haiku_text = generated_text.split("Haiku:")[1].strip()
        else:
            haiku_text = generated_text

        # Format the text as a proper haiku with 5-7-5 syllable structure
        haiku_lines = format_as_haiku(haiku_text)

        return "\n".join(haiku_lines)

    def get_response(self, user_input: str) -> str:
        """
        Generate a haiku response to user input.

        Args:
            user_input: The user's message

        Returns:
            A haiku response
        """
        # Occasionally change the season theme for variety
        if random.random() < 0.2:
            self.current_season = random.choice(list(self.seasons.keys()))

        # Prepare the input prompt
        prompt = self._prepare_input(user_input)

        # Generate a haiku
        haiku = self._generate_haiku(prompt)

        return haiku

    def display_formatted_haiku(self, haiku: str) -> None:
        """
        Display a haiku with nice formatting.

        Args:
            haiku: The haiku to display
        """
        lines = haiku.strip().split("\n")

        print("\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        for line in lines:
            padding = " " * (26 - len(line))
            print(f"┃  {line}{padding}┃")
        print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")


def interactive_mode():
    """
    Run the haiku bot in interactive mode where users can chat with it.
    """
    print("\n===================================")
    print("   Welcome to the Haiku Bot!")
    print("   Type 'exit' to end the chat")
    print("===================================\n")

    # Initialize the bot (in a real scenario, you would load your fine-tuned model)
    bot = HaikuBot()

    # Start conversation with a greeting
    greeting_haiku = (
        "Digital whispers\nNeural pathways awakened\nHaiku flows like streams"
    )
    bot.display_formatted_haiku(greeting_haiku)

    # Main chat loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            # Farewell haiku
            farewell = "Leaves fall from the tree\nOur conversation ends here\nUntil next meeting"
            print("\nHaiku Bot:")
            bot.display_formatted_haiku(farewell)
            break

        # Generate and display response
        try:
            haiku_response = bot.get_response(user_input)
            print("\nHaiku Bot:")
            bot.display_formatted_haiku(haiku_response)
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Let me try again with a simpler haiku.")
            simple_haiku = (
                "Words sometimes falter\nTechnology has limits\nTry once more, please"
            )
            bot.display_formatted_haiku(simple_haiku)


if __name__ == "__main__":
    print("Starting Haiku Bot...")
    print("Note: This is a prototype that uses the base GPT-2 model.")
    print("For better results, you should fine-tune the model on haiku data.")

    # Run in interactive mode
    interactive_mode()

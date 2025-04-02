# Module 7: Final Project – Building a Haiku-Generating Chatbot

> _Words flow like water,_  
> _Machine learns the art of breath,_  
> _Seasons in silicon._

## Overview

The journey from perceptron to transformer has led us to this final project: creating a chatbot that responds exclusively in haiku. This capstone experience brings together all the knowledge you've gained throughout the curriculum, demonstrating how neural networks have evolved from simple binary classifiers to systems capable of creative expression.

## Project Objectives

- Design and implement a haiku-generating chatbot
- Apply transformer architecture to language generation
- Fine-tune a pre-trained language model for a specific format
- Consider ethical implications of AI-generated creative content
- Create a user interface for interacting with your haiku bot

## Project Requirements

### Core Components

1. **Dataset Preparation**

   - Collect and curate a dataset of high-quality haiku
   - Process text data appropriately for the model
   - Create training, validation, and test splits

2. **Model Architecture**

   - Fine-tune a pre-trained language model (e.g., GPT-2 small)
   - Implement constraints to enforce the 5-7-5 syllable structure
   - Create a dialogue management system for the chatbot

3. **Training and Evaluation**

   - Train the model with appropriate hyperparameters
   - Evaluate quality using both automated metrics and human judgment
   - Implement techniques to ensure diversity in haiku generation

4. **User Interface**
   - Create a simple interface for users to interact with the chatbot
   - Display the generated haiku attractively
   - Provide mechanisms for user feedback

### Project Milestones

1. **Project Proposal** (Week 1)

   - Define scope, approach, and specific techniques
   - Preliminary research on haiku datasets and generation methods
   - Initial architecture design

2. **Data Collection and Processing** (Week 2)

   - Gather haiku corpus
   - Implement data processing pipeline
   - Prepare training data

3. **Model Implementation and Training** (Week 3-4)

   - Fine-tune model on haiku dataset
   - Implement syllable constraints
   - Create conversation management system

4. **Evaluation and Refinement** (Week 5)

   - Evaluate haiku quality and adherence to format
   - Gather feedback and iterate on the model
   - Implement improvements

5. **Final Presentation and Demo** (Week 6)
   - Create demonstration of working chatbot
   - Prepare presentation on approach and results
   - Write final report documenting the project

## Resources and Starting Points

### Code Template

The `/code` directory contains starter code for:

- `haiku_dataset.py` - Utilities for processing haiku data
- `model_finetuning.py` - Framework for fine-tuning a transformer
- `syllable_counter.py` - Functions for enforcing syllable structure
- `haiku_bot.py` - Simple interface for the chatbot

### Datasets

Some potential sources of haiku for your project:

- Classic Japanese haiku in translation
- Contemporary English haiku collections
- Existing haiku datasets on Kaggle or Hugging Face

### Papers and References

- "Constrained Language Generation Using Transformers"
- "Neural Poetry: Learning to Generate Poems with Controlled Meter and Rhyme"
- "Ethics of AI-Generated Creative Content"

## Evaluation Criteria

Your project will be evaluated on:

1. **Technical Implementation (40%)**

   - Correct implementation of the model architecture
   - Quality of code and documentation
   - Proper handling of training and evaluation

2. **Haiku Quality (30%)**

   - Adherence to haiku structure (5-7-5 syllables)
   - Semantic coherence and meaningfulness
   - Creative and poetic qualities

3. **Conversational Abilities (20%)**

   - Relevance of responses to user inputs
   - Coherence across multiple turns of dialogue
   - Handling of different conversation topics

4. **Presentation and Documentation (10%)**
   - Clear explanation of approach
   - Quality of demonstration
   - Completeness of documentation

## Connecting the Journey

From the humble perceptron to your haiku-generating chatbot, you've traversed the landscape of neural network architectures. Like a haiku poet who has mastered the form through years of practice, you've developed an understanding of the mathematical and computational principles that make these systems work.

The perceptron learned simple boundaries; your transformer model learns the boundaries of language and creativity. The convolutional networks found patterns in images; your model finds patterns in words. The recurrent networks captured sequences; your model captures the essence of poetic expression.

This final project is not just a technical exercise but a philosophical exploration of how far artificial intelligence has come, and what it means for machines to engage in creative expression that has been uniquely human for centuries.

May your chatbot's haiku be as elegant as the code that generates them!

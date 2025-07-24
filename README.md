# HR Q&A Bot

A Streamlit-based application that uses a fine-tuned Gemma 2B model to answer HR-related questions. The bot can match user questions with a database of HR Q&A pairs or generate responses using the fine-tuned model.

## Features

- Question matching using fuzzy search to find similar questions in the database
- Fine-tuned Gemma 2B model for generating HR-related responses
- Simple and intuitive Streamlit interface
- Chat history tracking

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Ananaya-3396/HR-QnA-Bot.git
   cd HR-QnA-Bot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run hr_streamlit_bot.py
   ```

## Project Structure

- `hr_streamlit_bot.py`: Main Streamlit application
- `HR questions.csv`: Database of HR questions and answers
- `requirements.txt`: Required Python packages
- `gemma_finetuned/`: Directory containing the fine-tuned model checkpoint

## Model Information

This project uses a fine-tuned version of Google's Gemma 2B model. The model was fine-tuned on HR-related Q&A data to provide accurate and helpful responses to HR questions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
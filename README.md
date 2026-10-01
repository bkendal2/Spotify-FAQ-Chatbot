# Spotify FAQ Chatbot

An AI-powered FAQ chatbot built with Python, LangChain, Ollama, and vector-based retrieval to answer common Spotify support questions.

## Project Overview

This project demonstrates how natural language processing and large language models can be used to build a practical customer support chatbot.

The chatbot retrieves relevant information from a collection of Spotify support FAQs and uses a locally running language model to generate helpful responses to user questions.

Rather than relying only on the language model's existing knowledge, the application uses retrieval to provide the model with relevant support information before generating its response.

## Technologies Used

- Python
- LangChain
- Ollama
- Llama 3.2
- Vector-based information retrieval
- Natural Language Processing (NLP)
- Large Language Models (LLMs)

## How It Works

1. Spotify FAQ information is stored as the chatbot's knowledge base.
2. User questions are processed through the retrieval system.
3. Relevant support information is identified and provided as context.
4. The local LLM uses that context to generate a response.
5. The chatbot displays the response along with the relevant Spotify support sources.

## Example Questions

The chatbot can respond to questions such as:

- Can I listen to Spotify offline?
- How do I reset my Spotify password?
- What should I do if I think my account was hacked?
- How do Spotify downloads work?
- What is included with Spotify Premium?

## Example Response

**User:** I think someone hacked my account. What should I do?

**SpotifyHelper:**

If you think your Spotify account was hacked, Spotify recommends resetting your password, signing out everywhere, and reviewing connected third-party apps. You should also use a strong password that you do not use for other services.

The chatbot then displays the relevant official Spotify support resources used to generate the answer.

## Project Files

- `spotify_faq_chatbot.py` - Main chatbot application
- `spotify_faq_chatbot_backup.py` - Backup version of the application

## Running the Project

Ollama must be installed and running locally.

The project uses the Llama 3.2 model through Ollama.

Run the chatbot with:

    python3 spotify_faq_chatbot.py

The application will launch an interactive command-line chatbot. Type your question and press Enter. Type `quit` to exit.

## What I Learned

This project gave me hands-on experience building an NLP application using modern generative AI tools. I learned how LangChain can be used to connect a language model with external information, how retrieval can provide relevant context before generating a response, and how a local LLM can be used instead of relying entirely on a paid cloud API.

I also gained experience troubleshooting LLM integrations and adapting the application to run locally with Ollama.

## Disclaimer

This project was created for educational and portfolio purposes and is not affiliated with or endorsed by Spotify.

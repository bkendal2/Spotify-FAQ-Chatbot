"""Spotify FAQ chatbot using LangChain, Ollama, and Chroma.

Install:
    pip install langchain-core langchain-ollama langchain-chroma

Required Ollama models:
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text

Run:
    python3 spotify_faq_chatbot.py
    python3 spotify_faq_chatbot.py --stats
    python3 spotify_faq_chatbot.py --ask "Can I listen to music offline?"

Educational prototype; Spotify does not endorse it.
"""

import argparse
import re
import sys
from datetime import date

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings


# ---------------------------------------------------------
# SPOTIFY FAQ DATASET
# ---------------------------------------------------------

# Each record contains four features:
# question, answer, source, and topic.
#
# The information was manually curated from official
# Spotify Support pages for this educational project.

FAQ = [
    {
        "question": "How do I cancel my Spotify Premium subscription?",
        "answer": "Go to your Spotify account page, open Manage your plan, and select Cancel subscription. Premium normally stays active until your next billing date, when your account switches to Spotify Free.",
        "source": "https://support.spotify.com/us/article/cancel-premium/",
        "topic": "billing",
    },
    {
        "question": "What happens to my playlists if I cancel Spotify Premium?",
        "answer": "You keep your playlists and saved music after your account switches to Spotify Free. You can still log in and listen with ads.",
        "source": "https://support.spotify.com/us/article/cancel-premium/",
        "topic": "billing",
    },
    {
        "question": "How do I change my Spotify Premium plan?",
        "answer": "If you pay Spotify directly, go to your account page, find Manage your subscription, select Change plan, and choose an available plan. The exact process can differ for Family, Duo, Google Play, or partner plans.",
        "source": "https://support.spotify.com/us/article/change-premium-plans/",
        "topic": "billing",
    },
    {
        "question": "What benefits come with Spotify Premium?",
        "answer": "Spotify Premium includes benefits such as ad-free music listening, music downloads for offline listening, greater playback control, and access to supported higher-quality audio features. Some benefits depend on the plan, device, or location.",
        "source": "https://support.spotify.com/us/article/your-premium-benefits/",
        "topic": "premium",
    },
    {
        "question": "Can I listen to Spotify music offline?",
        "answer": "Yes. Spotify Premium allows you to download albums, playlists, and podcasts for offline listening. Spotify Free users can download podcasts but not music for offline listening.",
        "source": "https://support.spotify.com/us/article/listen-offline/",
        "topic": "offline",
    },
    {
        "question": "How many Spotify songs can I download?",
        "answer": "Spotify allows Premium users to download as many as 10,000 tracks on each of up to 5 different devices.",
        "source": "https://support.spotify.com/us/article/listen-offline/",
        "topic": "offline",
    },
    {
        "question": "Do I need to go online to keep my Spotify downloads?",
        "answer": "Yes. You need to go online at least once every 30 days to keep your Spotify downloads.",
        "source": "https://support.spotify.com/us/article/listen-offline/",
        "topic": "offline",
    },
    {
        "question": "Can I download individual songs on Spotify?",
        "answer": "Spotify does not let you directly download an individual song by itself. You can add the song to a playlist and then download that playlist.",
        "source": "https://support.spotify.com/us/article/listen-offline/",
        "topic": "offline",
    },
    {
        "question": "Why aren't my Spotify downloads working?",
        "answer": "Check that you have an active internet connection, enough storage space, and have not reached the device limit. Spotify also suggests restarting the app and checking for cache-clearing or battery-saving apps that could interfere.",
        "source": "https://support.spotify.com/us/article/listen-offline/",
        "topic": "troubleshooting",
    },
    {
        "question": "Why did my Spotify downloads disappear?",
        "answer": "Downloads may be removed if you do not go online at least once every 30 days, reinstall Spotify, download on more than the allowed number of devices, use an outdated Spotify app, or have problems with an SD card.",
        "source": "https://support.spotify.com/us/article/listen-offline/",
        "topic": "troubleshooting",
    },
    {
        "question": "Why does Spotify say it is offline?",
        "answer": "Check whether Offline Mode is enabled and verify your internet connection. On mobile, Offline Mode can be found under Settings and privacy and then Data-saving and offline.",
        "source": "https://support.spotify.com/us/article/spotify-is-offline/",
        "topic": "troubleshooting",
    },
    {
        "question": "What should I do if Spotify cannot connect to the internet?",
        "answer": "Check whether other apps and websites can connect. Spotify suggests trying steps such as restarting Wi-Fi, restarting your router, checking network restrictions, trying another connection, and checking firewall settings.",
        "source": "https://support.spotify.com/us/article/spotify-is-offline/",
        "topic": "troubleshooting",
    },
    {
        "question": "How do I reset a forgotten Spotify password?",
        "answer": "Go to Spotify's password reset page, enter the email address linked to your Spotify account, and follow the instructions in the reset email.",
        "source": "https://support.spotify.com/us/article/reset-password/",
        "topic": "account",
    },
    {
        "question": "What should I do if I forgot my Spotify email or username?",
        "answer": "Go to Spotify's password reset page and try possible email addresses you own. Spotify sends a reset email only to an address registered to an account. You can also try the different login methods you may have used, such as email, phone number, Apple, or Google.",
        "source": "https://support.spotify.com/us/article/cannot-remember-login/",
        "topic": "account",
    },
    {
        "question": "What should I do if I think my Spotify account was hacked?",
        "answer": "Spotify recommends resetting your password, signing out everywhere, and reviewing connected third-party apps. You should use a strong password that you do not use for other services.",
        "source": "https://support.spotify.com/us/article/hacked-account-help/",
        "topic": "security",
    },
    {
        "question": "How do I sign out of Spotify everywhere?",
        "answer": "Go to your Spotify account page and select Sign out everywhere. Spotify notes that signing out everywhere can take up to one hour to take effect on all devices.",
        "source": "https://support.spotify.com/us/article/hacked-account-help/",
        "topic": "security",
    },
    {
        "question": "What is Spotify Connect?",
        "answer": "Spotify Connect lets you use one device to remotely control listening on another compatible device.",
        "source": "https://support.spotify.com/us/article/spotify-connect/",
        "topic": "devices",
    },
    {
        "question": "How do I use Spotify Connect?",
        "answer": "Open Spotify and play something, select the Connect device button, and choose the device you want to play on. When connecting to a speaker for the first time, the devices generally need to be on the same Wi-Fi network.",
        "source": "https://support.spotify.com/us/article/spotify-connect/",
        "topic": "devices",
    },
    {
        "question": "What should I do if Spotify Connect is not working?",
        "answer": "Spotify recommends restarting the Spotify app and connected device, making sure software is up to date, connecting devices to the same Wi-Fi, and restarting or changing the Wi-Fi connection.",
        "source": "https://support.spotify.com/us/article/spotify-connect/",
        "topic": "troubleshooting",
    },
    {
        "question": "Does Spotify Premium support lossless audio?",
        "answer": "Spotify Premium supports lossless music on compatible devices. Spotify states that lossless music can be available at up to 24-bit and 44.1 kHz. Availability and playback depend on compatible devices, content, settings, and network conditions.",
        "source": "https://support.spotify.com/us/article/lossless-audio-quality/",
        "topic": "audio",
    },
]


# ---------------------------------------------------------
# LANGCHAIN PROMPT
# ---------------------------------------------------------

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are SpotifyHelper, a Spotify FAQ assistant.

Answer the user's question using ONLY facts contained in the numbered Spotify FAQ records below.

Important instructions:
- Read all retrieved FAQ records before answering.
- You may paraphrase information from the records.
- Preserve all numbers, limits, dates, quantities, and restrictions exactly as stated in the FAQ records.
- Never replace a stated numerical limit with words such as "unlimited" or "no limit."
- If the FAQ says "10,000 tracks on each of up to 5 devices," you must preserve those numbers in your answer.
- Treat ordinary synonyms as equivalent when appropriate, such as "songs" and "tracks."
- If a retrieved record contains the factual answer, answer the question directly.

Spotify FAQ records:

{context}
""",
        ),
        ("human", "Question: {question}"),
    ]
)


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

def make_documents():
    """Clean FAQ text and convert records to LangChain Documents."""

    documents = []

    for index, row in enumerate(FAQ):

        # Normalize unnecessary whitespace.
        question = re.sub(r"\s+", " ", row["question"]).strip()
        answer = re.sub(r"\s+", " ", row["answer"]).strip()

        document = Document(
            page_content=(
                f"Question: {question}\n"
                f"Answer: {answer}"
            ),
            metadata={
                "source": row["source"],
                "topic": row["topic"],
                "id": index + 1,
            },
        )

        documents.append(document)

    return documents


# ---------------------------------------------------------
# VECTOR DATABASE
# ---------------------------------------------------------

def make_store(documents):
    """Create local embeddings and store them in Chroma."""

    print("Creating local embeddings...")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="spotify_faq_session",
    )

    print("Vector database ready.")

    return store


# ---------------------------------------------------------
# ANSWER USER QUESTIONS
# ---------------------------------------------------------

def answer(question, store, model):
    """Retrieve relevant FAQ records and generate an answer."""

    # Retrieve the three FAQ records most closely related
    # to the user's question and keep sufficiently relevant results.
    results = store.similarity_search_with_relevance_scores(
        question,
        k=min(3, len(FAQ))
    )

    matches = [doc for doc, score in results if score >= 0.35]

    if not matches:
        print("\nSpotifyHelper:")
        print("I don't have that information in my Spotify FAQ sources.")
        print("\nNo sufficiently relevant Spotify FAQ sources were found.")
        return

    # Give each retrieved document a temporary citation number.
    context = "\n\n".join(
        f"[{number}] {doc.page_content}\n"
        f"Source: {doc.metadata['source']}"
        for number, doc in enumerate(matches, 1)
    )

    # LangChain combines the prompt with the local LLM.
    chain = PROMPT | model

    message = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    response = (
        message.content
        if isinstance(message.content, str)
        else str(message.content)
    )

    print("\nSpotifyHelper:")
    print(response.strip())

    print("\nRetrieved official sources:")

    for number, doc in enumerate(matches, 1):
        print(
            f"[{number}] "
            f"{doc.metadata['source']}"
        )


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Ask questions about Spotify using "
            "official FAQ summaries."
        )
    )

    parser.add_argument(
        "--ask",
        help="Ask one question and exit"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset information"
    )

    args = parser.parse_args()

    documents = make_documents()

    # Dataset statistics can be viewed without starting
    # the language model.
    if args.stats:

        print(f"Records: {len(documents)}")
        print(
            "Fields: question (str), answer (str), "
            "source (URL str), topic (str)"
        )
        print(
            f"Source review date: "
            f"{date(2026, 9, 23).isoformat()}"
        )

        return

    try:

        # Create local Chroma vector database.
        store = make_store(documents)

        # Run Llama locally through Ollama.
        model = ChatOllama(
            model="llama3.2:3b",
            temperature=0
        )

        # Ask one question from the command line.
        if args.ask:

            answer(
                args.ask,
                store,
                model
            )

            return

        # Interactive chatbot.
        print("\nSpotify FAQ Chatbot")
        print("Powered locally by LangChain + Ollama")
        print("Type 'quit' to exit.")

        while True:

            try:
                question = input("\nYou: ").strip()

            except EOFError:
                break

            if question.lower() in {
                "quit",
                "exit"
            }:
                print("\nGoodbye!")
                break

            if question:

                answer(
                    question,
                    store,
                    model
                )

    except Exception as exc:

        print(
            f"Chatbot error: {exc}",
            file=sys.stderr
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
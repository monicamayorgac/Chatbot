# Biomedical Engineering Seminar Chatbot

A Chatbot designed to assist Biomedical Engineering students by analyzing seminar documentation, scientific papers, and course requirements.

The system uses **Google Gemini 2.0 Flash** for intelligence and **Local Embeddings** for privacy and performance.

##  Features

*  PDF Analysis: Upload multiple PDF documents (Seminar guidelines, scientific papers, etc.).
*  RAG Architecture: Retrieves specific chunks of text relevant to the user's question.
*  AI Intelligence: Uses Google's `gemini-flash-latest` model to generate context-aware answers.
*  Smart Citations: Provides the exact filename and page number for every answer to prevent hallucinations.
*  Local Embeddings: Uses `HuggingFace` local models to index data on your machine, avoiding extra API costs and rate limits.


##  Tech Stack

* **Python 3.10+**
* **Frontend:** Streamlit
* **Logic:** LangChain
* **Vector Database:** ChromaDB (Ephemeral/In-memory)
* **LLM:** Google Gemini API
* **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/monicamayorgac/Chatbot.git](https://github.com/monicamayorgac/Chatbot.git)
    cd Chatbot
    ```

2.  **Create a Virtual Environment (Optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Mac/Linux
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

##  Configuration

You need a Google AI Studio API Key to run this project.

1.  Get your key here: [Google AI Studio](https://aistudio.google.com/app/apikey)
2.  When running the app, enter the key in the sidebar.

##  Usage

Run the Streamlit application:

```bash
streamlit run seminar_bot.py

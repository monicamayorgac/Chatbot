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
    git clone https://github.com/monicamayorgac/Chatbot.git
    cd Chatbot
    ```

2.  ** Set up your environment(Optional but recommended):**
    
#### Option A: Using Conda (Recommended)

```bash
conda create -n 
conda activate

#### Option B: Using Virtual Environment (venv)

```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# venv\Scripts\activate   # On Windows

##  Configuration

You need a Google AI Studio API Key to run this project.
### Get your API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy it

```bash
# Create the secrets folder and file
mkdir -p .streamlit
echo 'GOOGLE_API_KEY = "your_api_key_here"' > .streamlit/secrets.toml
```

Replace `your_api_key_here` with your actual key.

##  Usage

Run the Streamlit application:
### Quick Start (Conda)

```bash
conda activate chatbot_project
streamlit run seminar_bot.py
```

### Quick Start (venv)

```bash
source venv/bin/activate
streamlit run seminar_bot.py
```

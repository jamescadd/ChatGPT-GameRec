# ChatGPT-GameRec
A GPT-powered chatbot that suggests games you'll love

## Setup & Install

### Install

```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows `cmd` or PowerShell, instead of using the `source` command above to activate the virtual environment, use:

```bash
env\Scripts\activate.bat
```

### Config

`cp config.template.json config.json`

Replace values from the template with your actual values:

- YouTube Data API v3 Key
- OpenAI API Key

## Execution

Run `python main.py -h` for more info and options.

### End-to-End

`python main.py -m end-to-end`

This extracts transcripts to a `.json` file, creates and saves FAISS
embeddings to the `faiss_index` directory, then starts an interactive
chat session with the extracted documents.

### Piecemeal

#### Extract Transcripts

`python main.py -m extract-transcipts`

Extracts transcripts to a `.json` file; defaults to `transcripts.json`, specify
with `-tf` argument.

#### Create Embeddings

`python main.py -m create-embeddings`

A `.json` file with extracted transcripts must exist to create embeddings;
specify with `-tf`, defaults to `transcripts.json`.

#### Summary Demo

Get an LLM-generated summary of one random video.

`python main.py -m summary-demo`

A `.json` file with extracted transcripts must exist to run the summary demo;
specify with `-tf`, defaults to `transcripts.json`.

#### Chat Demo

`python main.py -m chat-demo`

FAISS embeddings must have already been extracted and placed in the
`faiss_index` directory.

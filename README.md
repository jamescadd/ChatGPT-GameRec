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

#### VS Code & WSL with Ubuntu
Prerequisites: 
- Windows 11 or Windows 10 version 2004 or higher to support WSL2: [How to Install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- VS Code with WSL, Remote Development, and Python extensions

Recommended: [Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/install)
1. From a PowerShell prompt, run `wsl --install` from PowerShell and restart Windows
2. Install [Ubuntu](https://www.microsoft.com/store/productid/9PDXGNCFSCZV?ocid=pdpshare) from the Microsoft Store and setup a root user (a prompt should launch automatically after installing Ubuntu or by manually launching Ubuntu from Windows Terminal)
3. Clone the repository either in Windows (accessible in WSL via /mnt/c/<path to repo>) or WSL
4. Launch VS Code from the repository location in WSL: `code .`
5. Observe VS Code connected to WSL via green status bar icon at lower-right: `><WSL: Ubuntu`
6. Ensure python, pip, and venv are installed on Ubuntu. In VS Code Terminal or Windows Terminal: `apt install python3 python3-pip python3.10-venv`
7. Run the bash commands listed at the top of this section, beginning with `python3 -m venv env` (Note: python3 is used on Debian)

### Config

`cp config.template.json config.json`

Replace values from the template with your actual values:

- YouTube Data API v3 Key
- OpenAI API Key
- channel ID(s) (if desired)

**Note**: `channel_id` can be a single string for one channel ID, or a list
of strings if you want to extract video transcripts from multiple channels.

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

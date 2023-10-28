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
3. Clone the repository in the Linux filesystem for improved performance, either in Ubuntu shell or with Windows GitHub Desktop app referencing folder `\\wsl.localhost\Ubuntu\<path to repo>`.
4. Launch VS Code from the repository location in WSL: `code .` and observe VS Code connected to WSL via green status bar icon at lower-right: `><WSL: Ubuntu`
    1. Or: Launch VS Code from Windows, choose "Connect to WSL" in the command palette (Ctrl+Shift+P). Then open the repo folder in VS Code Explorer.
6. Ensure python, pip, and venv are installed on Ubuntu. In VS Code Terminal or Windows Terminal (Ubuntu): `apt install python3 python3-pip python3.10-venv`
    1. Note: Python 3.10 was the default version installed in Ubuntu 22 LTS as of this writing 9/4/2023.
7. In the VS Code command palette, run `Python: Create Environment`. This will create a new venv in the folder .venv by default.
    1. Alternatively, use the existing environment and run the command `Python: Create Terminal` after closing the existing terminal to use the .venv environment (terminal should then show the (.venv) prefix at the prompt).
8. After VS Code creates the environment, navigate to the VS Code Terminal window and install dependencies:
    1. Note that the terminal prompt is prefixed by `(.venv)` and you are working in the virtual environment created in VS Code
    2. Run `pip install --upgrade pip`
    3. Run `pip install -r requirements.txt`
9. VS Code setup is complete, `main.py` can be run in the debugger with F5 using the `Current File` configuration with `main.py` selected.
    1. The `Streamlit` debug configuration runs `streamlit run Chat.py`

### Config

`cp config.template.json config.json`

Replace values from the template with your actual values:

- YouTube Data API v3 Key
- OpenAI API Key
- channel ID(s) (if desired)

**Note**: `channel_id` can be a single string for one channel ID, or a list
of strings if you want to extract video transcripts from multiple channels.

#### Streamlit config

1. Create file: `.streamlit\secrets.toml`
2. Add OpenAI API key to secrets.toml: `OPENAI_API_KEY = "your_api_key"`

## Terminal Execution

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

## Streamlit execution

Run `streamlit run Chat.py` to start the Streamlit server

Browse to `http://localhost:8501` to view the app

### Debugging Streamlit

In VS Code, select the `Streamlit` configuration from the `Run and debug` dropdown to run `streamlit run Chat.py` in the VS Code Python debugger.
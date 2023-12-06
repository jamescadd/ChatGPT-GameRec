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

#### Docker install on Ubuntu
[Install docker using the repository](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)

#### VS Code & WSL with Ubuntu
Prerequisites: 
- Windows 11 or Windows 10 version 2004 or higher to support WSL2: [How to Install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- VS Code with WSL, Remote Development, and Python extensions

Optional: 
- [Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/install)
- [GitHub Desktop for Windows](https://desktop.github.com/)

> **Mind the End of Your Line ([link](https://adaptivepatchwork.com/2012/03/01/mind-the-end-of-your-line/)):** GitHub Desktop for Windows may change end of line characters from `LF` to `CRLF` to match Windows defaults. Scripts in this repo must use `LF` to run correctly in Docker - in particular run.sh will [fail with ^M](https://stackoverflow.com/questions/14219092/bash-script-bin-bashm-bad-interpreter-no-such-file-or-directory) if `CRLF` eol characters are used. In the file `C:\Users\[username]\.gitconfig` section labeled `[core]` add `autocrlf = false` to [preserve line endings in the repo](https://stackoverflow.com/questions/71995658/how-to-stop-github-desktop-from-changing-my-line-endings). Install the [VS Code extension eol](https://marketplace.visualstudio.com/items?itemName=sohamkamani.code-eol) for a visual indicator of `LF` and `CRLF` in the editor. A [`.gitattributes`](https://git-scm.com/docs/gitattributes) file in this repo prevents `CRLF` eol characters from being checked into `.sh` files.

1. From a PowerShell prompt, run `wsl --install` from PowerShell and restart Windows
2. Install [Ubuntu](https://www.microsoft.com/store/productid/9PDXGNCFSCZV?ocid=pdpshare) from the Microsoft Store and setup a root user (a prompt should launch automatically after installing Ubuntu or by manually launching Ubuntu from Windows Terminal)
3. Clone the repository in the Linux filesystem for improved performance, either in Ubuntu shell or with Windows GitHub Desktop app referencing folder `\\wsl.localhost\Ubuntu\<path to repo>`.
4. Launch VS Code from the repository location in WSL: `code .` and observe VS Code connected to WSL via green status bar icon at lower-right: `><WSL: Ubuntu`
    1. Or: Launch VS Code from Windows, choose "Connect to WSL" in the command palette (Ctrl+Shift+P). Then open the repo folder in VS Code Explorer.
6. Ensure python, pip, and venv are installed on Ubuntu. In VS Code Terminal or Windows Terminal (Ubuntu): `apt install python3 python3-pip python3.10-venv`
    1. Note: Python 3.10 was the default version installed in Ubuntu 22 LTS as of this writing 9/4/2023.
7. In the VS Code command palette, run `Python: Create Environment`. This will create a new venv in the folder .venv by default.
8. After VS Code creates the environment, navigate to the VS Code Terminal window and install dependencies:
    1. Note that the terminal prompt is prefixed by `(.venv)` and you are working in the virtual environment created in VS Code
    2. Run `pip install --upgrade pip`
    3. Run `pip install -r requirements.txt`
9. VS Code setup is complete, main.py can be run in the debugger with F5.

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

## Streamlit

### Command Line

```bash
streamlit run Chat.py
```

### Docker

```bash
docker build -t gamerec:v1 .
docker run --rm -p 8880:8501 gamerec:v1
```
Navigate to [http://localhost:8880](http://localhost:8880/) in your browser to view the application.
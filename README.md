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

## Execution

Run `python main.py -h` for more info and options.

### Extract Transcripts

If running for the first time, extract transcripts to a `.json` file:

`python main.py -c config.json -tf transcripts.json -et True -lt False`

### Load Transcripts from File

To load transcripts and get summaries from LLM, run:

`python main.py -c config.json -tf transcripts.json -lt True`

#!/bin/bash
nginx -t &&
service nginx start &&
streamlit run Chat.py --theme.base "dark"

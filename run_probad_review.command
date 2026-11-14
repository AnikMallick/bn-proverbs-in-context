#!/bin/bash

cd "$(dirname "$0")"

echo "========================================"
echo "      ProBad Dataset Reviewer"
echo "========================================"
echo

streamlit run app/review_app.py

echo
echo "ProBad has stopped."
read -p "Press Enter to close this window..."
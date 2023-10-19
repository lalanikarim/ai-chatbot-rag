# Streamlit + Langchain + Mistral: Retrieval Augmented Generation

Scrape a website for web content and pdfs and build a conversational ai chatbot from that knowledgebase.

*Instructions below are not complete.*

# TL;DR instructions

1. Install llama-cpp-python
2. Install langchain
3. Install streamlit
4. Download Mistral from HuggingFace from TheBloke's repo: mistral-7b-instruct-v0.1.Q4_0.gguf
5. Place model file in the `models` subfolder
6. Run streamlit

# Step by Step instructions

The setup assumes you have `python` already installed and `venv` module available.

1. Download the code or clone the repository.
2. Inside the root folder of the repository, initialize a python virtual environment:
```bash
python -m venv venv
```
3. Activate the python envitonment:
```bash
source venv/bin/activate
```
4. Install `langchain`, `llama.cpp`, and `streamlit`:
```bash
pip install langchain llama-cpp-python streamlit
```
5. Create a subdirectory to place the models in:
```bash
mkdir -p models
```
6. Download the `Mistral7b` quantized model from `huggingface` from the following link:
[mistral-7b-instruct-v0.1.Q4_0.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_0.gguf)
7. Start `streamlit`:
```bash
streamlit run main.py
```

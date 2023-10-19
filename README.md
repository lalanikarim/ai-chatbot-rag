# Streamlit + Langchain + LLama.cpp w/ Mistral: Retrieval Augmented Generation

Scrape a website for web content and pdfs and build a conversational ai chatbot from that knowledgebase.

This chatbot has conversational memory and can hold follow up conversations within the same session.

This code was tested on a WordPress Blog and as such has some logic that may not directly  
work on other websites. However, that can be easily fixed by making some tweaks and you will 
need to make some additional tweaks to suit your specific use cases.

Also, while this code can technically run on a computer without a GPU, running it on a GPU is recommended  
since the RAG process can be very slow otherwise.  

You will also need to change how you install `llama-cpp-python` package depending on your OS and whether you  
are planning on using a GPU or not.

# TL;DR instructions

1. Install llama-cpp-python
2. Install langchain
3. Install streamlit
4. Install beautifulsoup
5. Install PyMuPDF
6. Install sentence-transformers
7. Install docarray
8. Install pydantic 1.10.8
9. Download Mistral from HuggingFace from TheBloke's repo: mistral-7b-instruct-v0.1.Q5_0.gguf
10. Place model file in the `models` subfolder
11. Run streamlit

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
5. Install remaining requirements:
```bash
pip install beautifulsoup4 pymupdf sentence-transformers docarray pydantic==1.10.8
```
5. Create a subdirectory to place the models in:
```bash
mkdir -p models
```
6. Download the `Mistral7b` quantized model from `huggingface` from the following link:
[mistral-7b-instruct-v0.1.Q5_0.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_0.gguf)
7. Start `streamlit`:
```bash
streamlit run main.py
```

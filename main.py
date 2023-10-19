import streamlit as st
from bs4 import BeautifulSoup
import io
import fitz
import requests
from langchain.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter


# StreamHandler to intercept streaming output from the LLM.
# This makes it appear that the Language Model is "typing"
# in realtime.
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


@st.cache_data
def get_page_urls(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    links = [link['href'] for link in soup.find_all('a') if link['href'].startswith(url) and link['href'] not in [url]]
    links.append(url)
    return set(links)


def get_url_content(url):
    response = requests.get(url)
    if url.endswith('.pdf'):
        pdf = io.BytesIO(response.content)
        file = open('pdf.pdf', 'wb')
        file.write(pdf.read())
        file.close()
        doc = fitz.open('pdf.pdf')
        return (url, ''.join([text for page in doc for text in page.get_text()]))
    else:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Content containers. Here wordpress specific container css class name
        # used. This will be different for each website.
        content = soup.find_all('div', class_='wpb_content_element')
        text = [c.get_text().strip() for c in content if c.get_text().strip() != '']
        text = [line for item in text for line in item.split('\n') if line.strip() != '']

        # Post processing to exclude footer content.
        # This will be different for each website.
        arts_on = text.index('ARTS ON:')
        return (url, '\n'.join(text[:arts_on]))


@st.cache_resource
def get_retriever(urls):
    all_content = [get_url_content(url) for url in urls]
    documents = [Document(page_content=doc, metadata={'url': url}) for (url, doc) in all_content]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})
    return retriever


@st.cache_resource
def create_chain(_retriever):
    # A stream handler to direct streaming output on the chat screen.
    # This will need to be handled somewhat differently.
    # But it demonstrates what potential it carries.
    # stream_handler = StreamHandler(st.empty())

    # Callback manager is a way to intercept streaming output from the
    # LLM and take some action on it. Here we are giving it our custom
    # stream handler to make it appear as if the LLM is typing the
    # responses in real time.
    # callback_manager = CallbackManager([stream_handler])

    n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    llm = LlamaCpp(
            model_path="models/mistral-7b-instruct-v0.1.Q5_0.gguf",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=2048,
            # max_tokens=2048,
            temperature=0,
            # callback_manager=callback_manager,
            verbose=False,
            streaming=True,
            )

    # Template for the prompt.
    # template = "{question}"

    # We create a prompt from the template so we can use it with langchain
    # prompt = PromptTemplate(template=template, input_variables=["question"])

    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # We create a qa chain with our llm, retriever, and memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=_retriever, memory=memory, verbose=False
    )

    return qa_chain


# Set the webpage title
st.set_page_config(
    page_title="Your own AI-Chat!"
)

# Create a header element
st.header("Your own AI-Chat!")

# This sets the LLM's personality.
# The initial personality privided is basic.
# Try something interesting and notice how the LLM responses are affected.
# system_prompt = st.text_area(
#    label="System Prompt",
#    value="You are a helpful AI assistant who answers questions in short sentences.",
#    key="system_prompt")

if "base_url" not in st.session_state:
    st.session_state.base_url = ""

base_url = st.text_input("Enter the site url here", key="base_url")

if st.session_state.base_url != "":
    urls = get_page_urls(base_url)

    retriever = get_retriever(urls)

    # We store the conversation in the session state.
    # This will be used to render the chat conversation.
    # We initialize it with the first message we want to be greeted with.
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I help you today?"}
        ]

    if "current_response" not in st.session_state:
        st.session_state.current_response = ""

    # We loop through each message in the session state and render it as
    # a chat message.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # We initialize the quantized LLM from a local path.
    # Currently most parameters are fixed but we can make them
    # configurable.
    llm_chain = create_chain(retriever)

    # We take questions/instructions from the chat input to pass to the LLM
    if user_prompt := st.chat_input("Your message here", key="user_input"):

        # Add our input to the session state
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )

        # Add our input to the chat window
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Pass our input to the llm chain and capture the final responses.
        # It is worth noting that the Stream Handler is already receiving the
        # streaming response as the llm is generating. We get our response
        # here once the llm has finished generating the complete response.
        response = llm_chain.run(user_prompt)

        # Add the response to the session state
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        # Add the response to the chat window
        with st.chat_message("assistant"):
            st.markdown(response)

import warnings
import os
import datetime
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from chainlit.input_widget import TextInput
import datetime

# Suppress warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)

# Configure environment
os.environ["CHAINLIT_LANGUAGE"] = "en-US"
os.environ["CHAINLIT_NO_TRANSLATION"] = "true"
os.environ["CHAINLIT_HISTORY"] = "true"
os.environ["CHAINLIT_MAX_HISTORY"] = "20"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """
You are an expert in homeopathy. Based on the provided context, answer the user's question concisely and only provide the information requested.

Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, 
                          input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    retriever = db.as_retriever(search_kwargs={'k': 1})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=120,
        temperature=0.6,
        top_p=0.9
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def remove_redundant_text(text):
    """Remove duplicate lines while preserving order"""
    lines = text.split('\n')
    seen = set()
    clean_lines = []
    for line in lines:
        simplified = line.strip()
        if simplified and simplified not in seen:
            seen.add(simplified)
            clean_lines.append(line)
    return '\n'.join(clean_lines)

# @cl.on_chat_start function:
@cl.on_chat_start
async def start():
    # Initialize with proper history storage
    cl.user_session.set("message_history", [])
    
    # Correct TextInput configuration
    input_settings = TextInput(
        id="user_query_input",
        label="Medical Query",
        placeholder="Type your medical question here...",
        max_length=500,
        min_length=2
    )
    
    # Apply chat settings
    settings = cl.ChatSettings(
        input_settings=input_settings,
        show_clear_button=True
    )
    await settings.send()
    
    # Initialize bot
    chain = qa_bot()
    welcome_msg = "Hi, Welcome to Medical Bot. What is your query?"
    await cl.Message(content=welcome_msg).send()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    # Get history from session
    history = cl.user_session.get("message_history", [])
    
    # Store message with timestamp
    history.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "content": message.content
    })
    cl.user_session.set("message_history", history)
    
    # Process message
    chain = cl.user_session.get("chain")
    response = await chain.acall(
        message.content,
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    
    # Send response
    await cl.Message(content=response["result"]).send()


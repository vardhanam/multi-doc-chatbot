#pip install langchain pypdf openai chromadb tiktoken docx2txt unstructured 
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import shutil

# removing data from the earlier session

if os.path.exists("./data"):
    shutil.rmtree("./data")


os.environ["OPENAI_API_KEY"] = "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"

from langchain.document_loaders import UnstructuredEPubLoader


# Call the function to extract text from the PDF


documents = []
for file in os.listdir('docs'):
    epub_path = './docs/' + file 
    loader =   UnstructuredEPubLoader(epub_path)
    documents.extend(loader.load())


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

# we create our vectorDB, using the OpenAIEmbeddings tranformer to create
# embedding mugs from our text chunks. We set all the db information to be stored
# inside the ./data directory, so it doesn't clutter up our source files
vectordb = Chroma.from_documents(
  chunked_documents,
  embedding=OpenAIEmbeddings(),
  persist_directory='./data'
)
vectordb.persist()

#initiate the question and answer chain

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

qa_chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(),
    vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True
)

import sys

chat_history = []

while True:
    # this prints to the terminal, and waits to accept an input from the user
    query = input('Prompt: ')
    # give us a way to exit the script
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        sys.exit()
    # we pass in the query to the LLM, and print out the response. As well as
    # our query, the context of semantically relevant information from our
    # vector store will be passed in, as well as list of our chat history
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'])
    # we build up the chat_history list, based on our question and response
    # from the LLM, and the script then returns to the start of the loop
    # and is again ready to accept user input.
    chat_history.append((query, result['answer']))



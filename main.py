# import the ConversationSummaryBufferMemory, ConversationChain, ChatBedrock (BedrockChat) Langchain Modules
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_aws import ChatBedrock
from langchain_aws.llms.bedrock import Bedrock
from langchain_aws import BedrockLLM
from langchain_community.document_loaders import TextLoader

# Wrap within a function
def tao_index():
    # Define the data source and load data with TxtLoader
    data_load=TextLoader('tao.txt')
    # Split the Text based on Character, Tokens etc. - Recursively split by character - ["\n\n", "\n", " ", ""]
    data_split=RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100,chunk_overlap=10)
    # Create Embeddings -- Client connection
    data_embeddings=BedrockEmbeddings(
    credentials_profile_name= 'default',
    model_id='amazon.titan-embed-text-v1')
    # Create Vector DB, Store Embeddings and Index for Search - VectorstoreIndexCreator
    data_index=VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS)
    # Create index for HR Policy Document
    db_index=data_index.from_loaders([data_load])
    return db_index

# Write a function for invoking model- client connection with Bedrock with profile, model_id & Inference params- model_kwargs
def demo_chatbot():
    demo_llm=ChatBedrock(
       credentials_profile_name='default',
       model_id='anthropic.claude-3-haiku-20240307-v1:0',
       model_kwargs= {
           "max_tokens": 300,
           "temperature": 0.9,
           "top_p": 0.9,
           "stop_sequences": ["\n\nHuman:"]} )
    return demo_llm

#Test out the LLM with Predict method instead use invoke method
    #return demo_llm.invoke(input_text)
#response=demo_chatbot(input_text="Hi, what is the temperature in new york in January?")
#print(response)

# Write a function which searches the user prompt, searches the best match from Vector DB and sends both to LLM.
def tao_rag_response(index,question):
    rag_llm=demo_chatbot()
    tao_rag_query=index.query(question=question,llm=rag_llm)
    return tao_rag_query

# Create a Function for  ConversationSummaryBufferMemory  (llm and max token limit)
def demo_memory():
    llm_data=demo_chatbot()
    memory=ConversationSummaryBufferMemory(llm=llm_data,max_token_limit=300)
    return memory

# Create a Function for Conversation Chain - Input text + Memory
def demo_conversation(input_text,memory):
    llm_chain_data=demo_chatbot()
    llm_conversation=ConversationChain(llm=llm_chain_data,memory=memory,verbose=True)

    # Chat response using invoke (Prompt template)
    chat_reply=llm_conversation.invoke(input_text)
    return chat_reply['response']
# import streamlit and chatbot file
import streamlit as st 
import main as demo  # Import your Chatbot file as demo

# Set Title for Chatbot - https://docs.streamlit.io/library/api-reference/text/st.title
st.title("TAO Chatbot")  # Modify this based on the title you want

if 'vector_index' not in st.session_state: 
    with st.spinner("⏳ Wait for magic... All beautiful things in life take time :-)"):  # Spinner message
        st.session_state.vector_index = demo.tao_index()  # Your Index Function name from Backend File

# LangChain memory to the session cache - Session State - https://docs.streamlit.io/library/api-reference/session-state
if 'memory' not in st.session_state: 
    st.session_state.memory = demo.demo_memory()  # Modify the import and memory function() attributes initialize the memory

# Add the UI chat history to the session cache - Session State - https://docs.streamlit.io/library/api-reference/session-state
if 'chat_history' not in st.session_state:  # See if the chat history hasn't been created yet
    st.session_state.chat_history = []  # Initialize the chat history

# Re-render the chat history (Streamlit re-runs this script, so need this to preserve previous chat messages)
for message in st.session_state.chat_history: 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"]) 

# Enter the details for chatbot input box 
input_text = st.chat_input("Powered by Bedrock and Claude")  # Display a chat input box
if input_text: 
    with st.chat_message("user"): 
        st.markdown(input_text) 
    
    st.session_state.chat_history.append({"role": "user", "text": input_text}) 

    # Generate the chat response once
    response_content = demo.tao_rag_response(index=st.session_state.vector_index, question=input_text)  # Replace with RAG Function from backend file

    with st.chat_message("assistant"): 
        st.markdown(response_content) 
    
    st.session_state.chat_history.append({"role": "assistant", "text": response_content}) 
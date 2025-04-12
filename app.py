import streamlit as st
from barista import chat_graph, WELCOME_MSG
import os
from dotenv import load_dotenv

# Load environment variables (including your Google API Key)
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="BaristaBot ☕", page_icon=":coffee:")
st.title("BaristaBot ☕")
st.caption("An interactive café ordering assistant powered by Gemini + LangGraph")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "state" not in st.session_state:
    st.session_state.state = {"messages": [], "order": [], "finished": False}

# Display the welcome message once at the start
if not st.session_state.chat_history:
    st.chat_message("assistant").markdown(WELCOME_MSG)
    st.session_state.chat_history.append({"role": "assistant", "content": WELCOME_MSG})

# Display previous messages from chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Get user input
user_input = st.chat_input("What's your order?")
if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add to LangGraph state as dictionary message
    st.session_state.state["messages"].append({"role": "user", "content": user_input})

    # Invoke LangGraph flow
    response_state = chat_graph.invoke(st.session_state.state)

    # Update session state
    st.session_state.state = response_state

    # Extract latest response
    # Extract latest response
    latest_msg = response_state["messages"][-1]
    reply = getattr(latest_msg, "content", str(latest_msg))


    # Update history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(reply)

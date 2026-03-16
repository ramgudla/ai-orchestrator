from langchain_core.messages import HumanMessage
import streamlit as st
import asyncio
import logging
import time

from aio.core.agents import create_supervisor
from aio.util.utils import extract_ai_message_content

logging.basicConfig(level=logging.INFO)

async def chat_ui():

    """Main application entry point"""
    supervisor = create_supervisor()
    
    # Streamlit interface
    st.title("AI Assistant")
    logging.info("App started")

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI Assistant."}
        ]

    # Display all previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("How can I assist you today?"):

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            start_time = time.time()
            logging.info("Generating response...")
            with st.spinner("Processing..."):
                inputs = {
                    "messages": [
                        HumanMessage(
                            content=prompt
                        )
                    ],
                }
                config = {"configurable": {"thread_id": "1", "recursion_limit": 15}}   
                # Get the final step in the stream
                final_event = None
                async for step in supervisor.astream(inputs, config=config):
                    final_event = step  # Keep updating to the latest step
                    print(final_event)
                    print("\n")
                    response_message = extract_ai_message_content(final_event)
                    print(response_message)
                    
                    for agent, content in response_message:
                        assistant_reply = f"**Agent:** `{agent}`\n\n{content}"
                        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                        st.markdown(assistant_reply)
                        st.markdown("---")  


def main():
    asyncio.run(chat_ui())

if __name__ == "__main__":
    main()
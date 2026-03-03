from langchain_core.messages import HumanMessage
import streamlit as st
import asyncio
import logging
import time

from rgai.core.agents import create_supervisor, create_deepagent
from rgai.util.utils import extract_ai_message_content, parse_messages

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
                config = {"configurable": {"thread_id": "2", "recursion_limit": 15}}   
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

# ================================================= #
#            Use the supervisor
# ================================================= #

async def process_input(user_input):
    print("\nUser Request:", user_input)
    print("\nAI Response: ", end="", flush=True)  # Print label without newline and flush immediately

    # Use astream to get chunks
    async for msg, metadata in create_deepagent().astream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="messages"
    ):
        # Check if the message is from the AI
        from langchain_core.messages import AIMessageChunk
        if isinstance(msg, AIMessageChunk):
            # Print content token-by-token
            print(msg.content, end="", flush=True)  # Print content without newline and flush immediately
    
    # Alternatively, invoke the deepagent and print the final parsed result
    # result = await create_deepagent().ainvoke({"messages": [{"role": "user", "content": user_input}]})
    # parsed = parse_messages(result)
    # print(parsed)
      
    # ivoke the deepagent and print the streaming steps
    # async for step in create_supervisor().astream(
    # async for step in create_deepagent().astream(
    #     {"messages": [{"role": "user", "content": user_input}]}
    # ):
    #     # print(step)
    #     for update in step.values():
    #         print(update)
            # for message in update.get("messages", []):
            #     message.pretty_print()

    print("\n")

async def input_loop():
    """Continuously takes user input and processes it."""
    print("\nEnter text to process. Type 'exit' or 'quit' or 'q' or '!' to end the loop.")
    
    while True:
        try:
            # Take user input
            user_input = input("\nHow can I assist you today? ")
            
            # Check for exit conditions
            if user_input.lower() in ['exit', 'q', '!', 'quit']:
                print("\nExiting loop.")
                break
            
            # Process the input using a separate function
            await process_input(user_input)
            # print(result)
            
        except EOFError:
            # Handle cases where the input stream ends unexpectedly (e.g., piped input)
            print("\nEnd of input reached. Exiting.")
            break
        except KeyboardInterrupt:
            # Handle user interruption (e.g., Ctrl+C)
            print("\nUser interrupted. Exiting.")
            break

def entry_point():
    # asyncio.run(chat_ui())
    asyncio.run(input_loop())

if __name__ == "__main__":
    # asyncio.run(chat_ui())
    asyncio.run(input_loop())

# Example user queries to test the graph:
# calculate the result of 3 + 5
# Get the details of jira id RG-3552 from the RG project queue.
# Summarize the jira issue RG-457
# Summarize the jira issue EHRM-3552 from the jira-sd project queue.
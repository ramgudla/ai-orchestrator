import asyncio

from aio.core.agents import create_supervisor, create_deepagent
from aio.util.utils import parse_messages

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
    # print(result)
    # print(parsed)
    # print("\n")
      
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

def main():
    asyncio.run(input_loop())

if __name__ == "__main__":
    main()
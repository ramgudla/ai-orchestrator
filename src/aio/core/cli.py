#!/usr/bin/env python3
"""
CLI for Streaming Deep Agents
Simple command-line interface to interact with streaming agents
"""

import asyncio
import argparse
import sys
import os
from typing import Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
import signal

# Import our streaming agents
from aio.core.streaming_deep_agents import (
    StreamingDeepAgent,
    StreamingAgentOrchestrator,
    AgentRole,
    Task,
    TaskStatus,
)

# Load environment variables
load_dotenv()

console = Console()


class StreamingAgentsCLI:
    """Command Line Interface for Streaming Deep Agents"""

    def __init__(self):
        self.orchestrator = StreamingAgentOrchestrator()
        self.running = True

        # Setup signal handler for graceful exit
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        console.print("\n[yellow]Exiting gracefully...[/yellow]")
        self.running = False
        sys.exit(0)

    def display_welcome(self):
        """Display welcome message"""
        welcome_text = """
# Streaming Deep Agents CLI

Welcome to the Streaming Deep Agents command-line interface!
This example implementation allows you to interact with AI agents that can stream responses in real-time.

## Available Commands:
- **chat**: Chat with a specific agent
- **task**: Process a complex task with multiple agents
- **list**: List all available agents
- **add**: Add a new agent
- **help**: Show this help message
- **exit**: Exit the CLI
        """
        console.print(Markdown(welcome_text))

    def display_agents(self):
        """Display list of available agents"""
        if not self.orchestrator.agents:
            console.print(
                "[yellow]No agents available. Please add agents first.[/yellow]"
            )
            return

        console.print("\n[bold cyan]Available Agents:[/bold cyan]")
        for name, agent in self.orchestrator.agents.items():
            console.print(f"  • {name} - [dim]{agent.role.value}[/dim]")

    async def chat_with_agent(self):
        """Interactive chat with a specific agent"""
        self.display_agents()

        if not self.orchestrator.agents:
            return

        agent_name = Prompt.ask(
            "\n[cyan]Which agent would you like to chat with?[/cyan]"
        )

        if agent_name not in self.orchestrator.agents:
            console.print(f"[red]Agent '{agent_name}' not found.[/red]")
            return

        agent = self.orchestrator.agents[agent_name]
        console.print(f"\n[green]Starting chat with {agent_name}...[/green]")
        console.print("[dim]Type 'exit' to end the chat[/dim]\n")

        while True:
            user_input = Prompt.ask(f"[bold blue]You[/bold blue]")

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Ending chat...[/yellow]")
                break

            console.print(f"\n[bold cyan]{agent_name}[/bold cyan]: ", end="")

            # Stream the response
            response = ""
            async for token in agent.stream_response(user_input):
                console.print(token, end="")
                response += token

            console.print("\n")

    async def process_complex_task(self):
        """Process a complex task using multiple agents"""
        task_description = Prompt.ask("\n[cyan]Describe the complex task[/cyan]")

        if not self.orchestrator.agents:
            console.print(
                "[yellow]No agents available. Creating standard agents...[/yellow]"
            )
            self.orchestrator.create_standard_agents()

        console.print("\n[green]Processing complex task...[/green]\n")
        await self.orchestrator.process_complex_task(task_description)

    def add_agent(self):
        """Add a new agent to the orchestrator"""
        console.print("\n[bold]Add New Agent[/bold]")

        name = Prompt.ask("Agent name")

        console.print("\nAvailable roles:")
        for role in AgentRole:
            console.print(f"  • {role.value}")

        role_str = Prompt.ask("Agent role", choices=[r.value for r in AgentRole])
        role = AgentRole(role_str)

        model = Prompt.ask("Model name", default="gpt-4.1-mini")
        temperature = float(Prompt.ask("Temperature (0.0-1.0)", default="0.7"))
        streaming = Confirm.ask("Enable streaming?", default=True)

        # Create and add the agent
        agent = StreamingDeepAgent(
            name=name,
            role=role,
            model_name=model,
            temperature=temperature,
            streaming=streaming,
        )

        self.orchestrator.add_agent(agent)
        console.print(f"[green]Agent '{name}' added successfully![/green]")

    def setup_standard_agents(self):
        """Setup standard set of agents"""
        console.print("\n[yellow]Setting up standard agents...[/yellow]")
        self.orchestrator.create_standard_agents()
        console.print("[green]Standard agents created successfully![/green]")

    async def run_interactive_mode(self):
        """Run the CLI in interactive mode"""
        self.display_welcome()

        # Ask if user wants to create standard agents
        if Confirm.ask(
            "\n[cyan]Would you like to create the standard agent set?[/cyan]",
            default=True,
        ):
            self.setup_standard_agents()

        while self.running:
            console.print("\n" + "=" * 50)
            command = Prompt.ask(
                "\n[bold green]Command[/bold green]",
                choices=["chat", "task", "list", "add", "help", "exit"],
                default="help",
            )

            try:
                if command == "chat":
                    await self.chat_with_agent()
                elif command == "task":
                    await self.process_complex_task()
                elif command == "list":
                    self.display_agents()
                elif command == "add":
                    self.add_agent()
                elif command == "help":
                    self.display_welcome()
                elif command == "exit":
                    if Confirm.ask(
                        "[yellow]Are you sure you want to exit?[/yellow]", default=False
                    ):
                        console.print("[green]Goodbye![/green]")
                        self.running = False
                        break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    async def run_single_command(self, args):
        """Run a single command from command line arguments"""
        if args.command == "chat":
            if args.agent and args.prompt:
                # Direct chat without interactive mode
                if args.agent not in self.orchestrator.agents:
                    # Create the agent if it doesn't exist
                    role = AgentRole.COORDINATOR  # Default role
                    agent = StreamingDeepAgent(name=args.agent, role=role)
                    self.orchestrator.add_agent(agent)

                agent = self.orchestrator.agents[args.agent]
                console.print(f"[cyan]{args.agent}:[/cyan] ", end="")

                async for token in agent.stream_response(args.prompt):
                    console.print(token, end="")
                console.print()
            else:
                await self.chat_with_agent()

        elif args.command == "task":
            if args.prompt:
                if not self.orchestrator.agents:
                    self.orchestrator.create_standard_agents()
                await self.orchestrator.process_complex_task(args.prompt)
            else:
                await self.process_complex_task()

        elif args.command == "list":
            self.display_agents()

        elif args.command == "setup":
            self.setup_standard_agents()


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(
        description="Streaming Deep Agents CLI - Interactive AI agents with real-time streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["chat", "task", "list", "setup", "interactive"],
        default="interactive",
        help="Command to execute (default: interactive)",
    )

    parser.add_argument("-a", "--agent", help="Agent name for chat command")

    parser.add_argument("-p", "--prompt", help="Prompt or task description")

    parser.add_argument(
        "--no-streaming", action="store_true", help="Disable streaming output"
    )

    args = parser.parse_args()

    # Check for OpenAI API key
    # if not os.getenv("OPENAI_API_KEY"):
    #     console.print("[red]Error: OPENAI_API_KEY not found![/red]")
    #     console.print("Please set your OpenAI API key:")
    #     console.print("  export OPENAI_API_KEY='your-key-here'")
    #     console.print("  or create a .env file with OPENAI_API_KEY=your-key-here")
    #     sys.exit(1)

    # Create CLI instance
    cli = StreamingAgentsCLI()

    # Run the appropriate mode
    if args.command == "interactive" or args.command is None:
        asyncio.run(cli.run_interactive_mode())
    else:
        asyncio.run(cli.run_single_command(args))


if __name__ == "__main__":
    main()

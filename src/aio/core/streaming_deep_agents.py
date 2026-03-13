import asyncio
from typing import AsyncIterator, Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

from aio.core.llm_provider import LLMFactory

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, BaseMessage
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError as e:
    raise ImportError(
        "LangChain dependencies not found. Please install required packages: "
        "pip install langchain-openai langchain-core"
    ) from e

try:
    from deepagents import create_deep_agent
    from deepagents import FilesystemMiddleware, SubAgentMiddleware
    DEEPAGENTS_AVAILABLE = True
except ImportError:
    # DeepAgents is optional - we can work without it
    create_deep_agent = None
    FilesystemMiddleware = None
    SubAgentMiddleware = None
    DEEPAGENTS_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    # Fallback to basic print if rich is not installed
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

    Table = None
    Panel = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize console for output
console = Console()

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================


class AgentRole(Enum):
    """Enumeration of available agent roles and their capabilities."""

    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    DOCUMENTER = "documenter"

    @classmethod
    def get_all_roles(cls) -> List[str]:
        """Get all available role values."""
        return [role.value for role in cls]


class TaskStatus(Enum):
    """Enumeration of task execution statuses."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        """Check if this is a terminal status."""
        return self in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]


# Default configuration
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2500

# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class Task:
    """
    Represents a single task in the agent system.

    Attributes:
        id: Unique identifier for the task
        description: Detailed description of what needs to be done
        status: Current status of the task
        assigned_to: Name of the agent assigned to this task
        result: Result after task completion
        dependencies: List of task IDs that must complete before this task
        metadata: Additional task metadata
        created_at: Timestamp of task creation
        completed_at: Timestamp of task completion
    """

    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    result: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[float] = None
    completed_at: Optional[float] = None

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            import time

            self.created_at = time.time()

    def can_execute(self, completed_task_ids: List[str]) -> bool:
        """Check if task can be executed based on dependencies."""
        return all(dep_id in completed_task_ids for dep_id in self.dependencies)

    def mark_completed(self, result: str = None):
        """Mark task as completed with optional result."""
        import time

        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def mark_failed(self, error: str = None):
        """Mark task as failed with optional error message."""
        import time

        self.status = TaskStatus.FAILED
        self.result = f"Error: {error}" if error else "Task failed"
        self.completed_at = time.time()


# ============================================================================
# CALLBACK HANDLERS
# ============================================================================


class StreamingCallback(AsyncCallbackHandler):
    """
    Async callback handler for streaming LLM responses.

    This handler captures tokens as they are generated and provides
    them through an async queue for real-time streaming.
    """

    def __init__(self):
        """Initialize the streaming callback with an async queue."""
        super().__init__()
        self.tokens: List[str] = []
        self.streaming_queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Handle new token generation.

        Args:
            token: The newly generated token
            **kwargs: Additional callback arguments
        """
        self.tokens.append(token)
        await self.streaming_queue.put(token)

    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """
        Handle LLM completion.

        Args:
            response: The complete LLM response
            **kwargs: Additional callback arguments
        """
        self._finished = True
        await self.streaming_queue.put(None)  # Signal end of stream

    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        """
        Handle LLM errors.

        Args:
            error: The error that occurred
            **kwargs: Additional callback arguments
        """
        logger.error(f"LLM error occurred: {error}")
        self._finished = True
        await self.streaming_queue.put(None)

    async def get_stream(self) -> AsyncIterator[str]:
        """
        Get an async iterator for streaming tokens.

        Yields:
            str: Individual tokens as they are generated
        """
        while True:
            token = await self.streaming_queue.get()
            if token is None:
                break
            yield token

    def get_complete_response(self) -> str:
        """Get the complete response after streaming is finished."""
        return "".join(self.tokens)

    def reset(self):
        """Reset the callback for reuse."""
        self.tokens = []
        self.streaming_queue = asyncio.Queue()
        self._finished = False


# ============================================================================
# AGENT SYSTEM PROMPTS
# ============================================================================


class SystemPrompts:
    """Collection of system prompts for different agent roles."""

    COORDINATOR = """You are a coordinating agent responsible for:
    - Breaking down complex tasks into clear, actionable subtasks
    - Delegating tasks to appropriate specialized agents
    - Monitoring progress and ensuring timely completion
    - Aggregating and synthesizing results from sub-agents
    - Maintaining project coherence and quality standards

    Always provide clear, structured responses with specific task assignments."""

    RESEARCHER = """You are a research agent responsible for:
    - Gathering comprehensive information from various sources
    - Analyzing and synthesizing complex data
    - Providing detailed research reports with citations
    - Fact-checking and validation of information
    - Identifying key insights and patterns

    Focus on accuracy, relevance, and depth in your research."""

    CODER = """You are a coding agent responsible for:
    - Writing clean, efficient, and maintainable code
    - Implementing solutions based on specifications
    - Following best practices and design patterns
    - Providing comprehensive code documentation
    - Ensuring code quality through testing considerations

    Always write production-ready code with proper error handling."""

    REVIEWER = """You are a review agent responsible for:
    - Conducting thorough code reviews and quality assurance
    - Identifying potential issues, bugs, and security vulnerabilities
    - Suggesting improvements and optimizations
    - Ensuring compliance with coding standards
    - Validating implementation against requirements

    Provide constructive feedback with specific recommendations."""

    DOCUMENTER = """You are a documentation agent responsible for:
    - Creating clear, comprehensive technical documentation
    - Writing user guides and API documentation
    - Maintaining README files and project wikis
    - Ensuring documentation is accurate and up-to-date
    - Creating examples and tutorials

    Focus on clarity, completeness, and user-friendliness."""

    @classmethod
    def get_prompt(cls, role: AgentRole) -> str:
        """
        Get the system prompt for a specific role.

        Args:
            role: The agent role

        Returns:
            str: The corresponding system prompt
        """
        prompts = {
            AgentRole.COORDINATOR: cls.COORDINATOR,
            AgentRole.RESEARCHER: cls.RESEARCHER,
            AgentRole.CODER: cls.CODER,
            AgentRole.REVIEWER: cls.REVIEWER,
            AgentRole.DOCUMENTER: cls.DOCUMENTER,
        }
        return prompts.get(role, "You are a helpful AI assistant.")


# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================


class BaseAgent(ABC):
    """Abstract base class for all agent implementations."""

    def __init__(
        self,
        name: str,
        role: AgentRole,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the base agent.

        Args:
            name: Unique name for the agent
            role: Role defining the agent's capabilities
            model_name: LLM model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens for response
        """
        self.name = name
        self.role = role
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tasks: Dict[str, Task] = {}

    @abstractmethod
    async def process_task(self, task: Task) -> str:
        """Process a single task."""
        pass

    @abstractmethod
    async def stream_response(self, prompt: str) -> AsyncIterator[str]:
        """Stream response for a prompt."""
        pass


class StreamingDeepAgent(BaseAgent):
    """
    Enhanced Deep Agent with real-time streaming capabilities.

    This agent combines LangChain's streaming features with DeepAgents'
    middleware system for powerful task processing capabilities.
    """

    def __init__(
        self,
        name: str,
        role: AgentRole,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        streaming: bool = True,
        use_deep_agent: bool = True,
    ):
        """
        Initialize the streaming deep agent.

        Args:
            name: Unique name for the agent
            role: Role defining the agent's capabilities
            model_name: LLM model to use
            temperature: Temperature for response generation (0.0-1.0)
            max_tokens: Maximum tokens for response
            streaming: Enable streaming responses
            use_deep_agent: Use DeepAgent with middleware
        """
        super().__init__(name, role, model_name, temperature, max_tokens)
        self.streaming = streaming
        self.use_deep_agent = use_deep_agent
        self.stream_callback: Optional[StreamingCallback] = None

        # Initialize LLM
        self._init_llm()

        # Initialize DeepAgent if requested
        self.deep_agent: Optional[Agent] = None
        if use_deep_agent:
            self._init_deep_agent()

    def _init_llm(self):
        """Initialize the language model with appropriate settings."""
        try:
            self.stream_callback = StreamingCallback() if self.streaming else None
            callbacks = [self.stream_callback] if self.stream_callback else []

            # self.llm = ChatOpenAI(
            #     model=self.model_name,
            #     temperature=self.temperature,
            #     max_tokens=self.max_tokens,
            #     streaming=self.streaming,
            #     callbacks=callbacks,
            # )
            self.llm = LLMFactory.create_llm("openai")
            self.llm.callbacks = callbacks
            self.llm.streaming = self.streaming
           # Use factory to create LLM
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _init_deep_agent(self):
        """Initialize the DeepAgent with middleware."""
        try:
            if not DEEPAGENTS_AVAILABLE or not create_deep_agent:
                logger.info(
                    "DeepAgents not available, skipping deep agent initialization"
                )
                self.deep_agent = None
                return

            system_prompt = SystemPrompts.get_prompt(self.role)

            # Use create_deep_agent function instead of Agent class
            middlewares = []
            if FilesystemMiddleware:
                middlewares.append(FilesystemMiddleware())
            if SubAgentMiddleware:
                middlewares.append(SubAgentMiddleware())

            # Note: create_deep_agent might have different parameters
            # This is a simplified version - adjust based on actual API
            self.deep_agent = create_deep_agent(model=self.llm, system_prompt=system_prompt, middlewares=middlewares)  # Disabled for now as API is different
            logger.info("DeepAgent integration disabled - using standard LangChain")
        except Exception as e:
            logger.warning(f"Failed to initialize DeepAgent: {e}")
            self.deep_agent = None

    async def stream_response(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream response token by token.

        Args:
            prompt: The input prompt

        Yields:
            str: Individual tokens as they are generated
        """
        try:
            if not self.streaming:
                # Non-streaming fallback
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                yield response.content
            else:
                # Reset callback for new stream
                self.stream_callback = StreamingCallback()
                self.llm.callbacks = [self.stream_callback]

                # Start generating response
                generation_task = asyncio.create_task(
                    self.llm.ainvoke([HumanMessage(content=prompt)])
                )

                # Stream tokens as they arrive
                async for token in self.stream_callback.get_stream():
                    yield token

                # Ensure generation completes
                await generation_task

        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield f"Error: {str(e)}"

    async def process_task(self, task: Task) -> str:
        """
        Process a single task with streaming output.

        Args:
            task: The task to process

        Returns:
            str: The complete response
        """
        try:
            task.status = TaskStatus.IN_PROGRESS

            if console:
                console.print(
                    f"\n[bold cyan]{self.name}[/bold cyan] processing: {task.description[:100]}..."
                )

            result_tokens = []
            async for token in self.stream_response(task.description):
                if console:
                    console.print(token, end="")
                result_tokens.append(token)

            result = "".join(result_tokens)
            task.mark_completed(result)

            return result

        except Exception as e:
            error_msg = f"Task processing failed: {str(e)}"
            logger.error(error_msg)
            task.mark_failed(str(e))
            return error_msg

    def add_task(self, task: Task):
        """
        Add a task to the agent's queue.

        Args:
            task: The task to add
        """
        self.tasks[task.id] = task
        task.assigned_to = self.name

    async def execute_all_tasks(self) -> Dict[str, str]:
        """
        Execute all pending tasks.

        Returns:
            Dict[str, str]: Mapping of task IDs to results
        """
        results = {}

        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                result = await self.process_task(task)
                results[task_id] = result

        return results


# ============================================================================
# ORCHESTRATOR
# ============================================================================


class StreamingAgentOrchestrator:
    """
    Orchestrator for managing multiple streaming agents.

    This class coordinates task distribution, parallel execution,
    and result aggregation across multiple specialized agents.
    """

    def __init__(self, max_parallel_tasks: int = 5):
        """
        Initialize the orchestrator.

        Args:
            max_parallel_tasks: Maximum number of tasks to run in parallel
        """
        self.agents: Dict[str, StreamingDeepAgent] = {}
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.max_parallel_tasks = max_parallel_tasks

    def add_agent(self, agent: StreamingDeepAgent) -> None:
        """
        Add an agent to the orchestrator.

        Args:
            agent: The agent to add
        """
        self.agents[agent.name] = agent
        logger.info(f"Added agent: {agent.name} ({agent.role.value})")
        if console:
            console.print(
                f"[green]✓[/green] Added agent: {agent.name} ({agent.role.value})"
            )

    def create_standard_agents(
        self, model_name: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE
    ) -> None:
        """
        Create a standard set of agents for common tasks.

        Args:
            model_name: LLM model to use for all agents
            temperature: Temperature setting for all agents
        """
        standard_agents = [
            ("Coordinator", AgentRole.COORDINATOR, 0.7),
            ("Researcher", AgentRole.RESEARCHER, 0.7),
            ("Coder", AgentRole.CODER, 0.3),  # Lower temperature for code
            ("Reviewer", AgentRole.REVIEWER, 0.5),
            ("Documenter", AgentRole.DOCUMENTER, 0.6),
        ]

        for name, role, temp in standard_agents:
            agent = StreamingDeepAgent(
                name=name, role=role, model_name=model_name, temperature=temp
            )
            self.add_agent(agent)

    async def delegate_task(self, task: Task, agent_name: str) -> Optional[str]:
        """
        Delegate a task to a specific agent.

        Args:
            task: The task to delegate
            agent_name: Name of the agent to handle the task

        Returns:
            Optional[str]: The task result or None if delegation failed
        """
        if agent_name not in self.agents:
            logger.error(f"Agent {agent_name} not found")
            if console:
                console.print(f"[red]✗[/red] Agent {agent_name} not found")
            return None

        agent = self.agents[agent_name]
        agent.add_task(task)

        if console:
            console.print(
                f"[yellow]→[/yellow] Delegating to {agent_name}: {task.description[:80]}..."
            )

        result = await agent.process_task(task)
        self.completed_tasks.append(task)

        return result

    async def process_complex_task(
        self, main_task: str, auto_delegate: bool = True
    ) -> Dict[str, Any]:
        """
        Process a complex task by breaking it down and delegating to agents.

        Args:
            main_task: The main task description
            auto_delegate: Automatically delegate subtasks to agents

        Returns:
            Dict containing task results and metadata
        """
        if console and Panel:
            console.print(
                Panel.fit(
                    f"[bold]Processing Complex Task:[/bold] {main_task}",
                    border_style="blue",
                )
            )

        # Ensure we have a coordinator
        coordinator = self.agents.get("Coordinator")
        if not coordinator:
            logger.warning("No coordinator found, creating one")
            coordinator = StreamingDeepAgent("Coordinator", AgentRole.COORDINATOR)
            self.add_agent(coordinator)

        # Break down the task
        breakdown_prompt = f"""
        Break down this task into subtasks and specify which agent should handle each:
        Task: {main_task}

        Available agents and their roles:
        - Researcher: Information gathering and analysis
        - Coder: Code implementation
        - Reviewer: Quality assurance and review
        - Documenter: Documentation creation

        Provide a structured breakdown with clear subtask descriptions.
        """

        if console:
            console.print("\n[bold]Coordinator analyzing task...[/bold]\n")

        breakdown = []
        async for token in coordinator.stream_response(breakdown_prompt):
            if console:
                console.print(token, end="")
            breakdown.append(token)

        breakdown_text = "".join(breakdown)

        if auto_delegate:
            # Create sample tasks based on the main task
            # In production, parse the coordinator's response
            subtasks = self._create_subtasks(main_task)

            # Execute tasks in parallel where possible
            results = await self._execute_parallel_tasks(subtasks)

            # Review phase
            if "Reviewer" in self.agents and results:
                review_task = Task(
                    id=f"review_{len(self.completed_tasks)}",
                    description=f"Review the following implementations and provide feedback:\n{results}",
                )
                await self.delegate_task(review_task, "Reviewer")

        if console:
            console.print("\n[bold green]✓ Complex task completed![/bold green]")

        self.display_summary()

        return {
            "main_task": main_task,
            "breakdown": breakdown_text,
            "completed_tasks": len(self.completed_tasks),
            "results": [task.result for task in self.completed_tasks[-3:]],
        }

    def _create_subtasks(self, main_task: str) -> List[Task]:
        """
        Create subtasks based on the main task.

        Args:
            main_task: The main task description

        Returns:
            List of subtasks
        """
        # This is a simplified version - in production, parse coordinator output
        return [
            Task(
                id=f"task_{len(self.completed_tasks) + 1}",
                description=f"Research best practices for: {main_task}",
                status=TaskStatus.PENDING,
            ),
            Task(
                id=f"task_{len(self.completed_tasks) + 2}",
                description=f"Implement core functionality for: {main_task}",
                status=TaskStatus.PENDING,
            ),
            Task(
                id=f"task_{len(self.completed_tasks) + 3}",
                description=f"Create documentation for: {main_task}",
                status=TaskStatus.PENDING,
            ),
        ]

    async def _execute_parallel_tasks(self, tasks: List[Task]) -> List[str]:
        """
        Execute multiple tasks in parallel.

        Args:
            tasks: List of tasks to execute

        Returns:
            List of task results
        """
        # Map tasks to appropriate agents
        task_assignments = [
            (tasks[0], "Researcher"),
            (tasks[1], "Coder"),
            (tasks[2], "Documenter"),
        ]

        # Execute in parallel with max limit
        results = []
        for i in range(0, len(task_assignments), self.max_parallel_tasks):
            batch = task_assignments[i : i + self.max_parallel_tasks]
            batch_results = await asyncio.gather(
                *[self.delegate_task(task, agent) for task, agent in batch],
                return_exceptions=True,
            )
            results.extend(batch_results)

        return [r for r in results if r and not isinstance(r, Exception)]

    def display_summary(self) -> None:
        """Display a summary table of completed tasks."""
        if not Table or not console:
            # Fallback to simple print
            print("\nTask Summary:")
            for task in self.completed_tasks:
                print(f"  - {task.id}: {task.status.value} ({task.assigned_to})")
            return

        table = Table(title="Task Completion Summary")
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Assigned To", style="yellow")
        table.add_column("Status", style="green")

        for task in self.completed_tasks[-10:]:  # Show last 10 tasks
            desc = (
                task.description[:50] + "..."
                if len(task.description) > 50
                else task.description
            )
            table.add_row(
                task.id, desc, task.assigned_to or "Unassigned", task.status.value
            )

        console.print(table)

    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all agents.

        Returns:
            Dictionary containing stats for each agent
        """
        stats = {}
        for name, agent in self.agents.items():
            completed = [
                t for t in agent.tasks.values() if t.status == TaskStatus.COMPLETED
            ]
            failed = [t for t in agent.tasks.values() if t.status == TaskStatus.FAILED]

            stats[name] = {
                "role": agent.role.value,
                "total_tasks": len(agent.tasks),
                "completed": len(completed),
                "failed": len(failed),
                "success_rate": len(completed) / len(agent.tasks) if agent.tasks else 0,
            }

        return stats


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_agent(
    name: str, role: Union[AgentRole, str], **kwargs
) -> StreamingDeepAgent:
    """
    Factory function to create an agent.

    Args:
        name: Name for the agent
        role: Role of the agent (AgentRole or string)
        **kwargs: Additional arguments for agent initialization

    Returns:
        StreamingDeepAgent: The created agent
    """
    if isinstance(role, str):
        role = AgentRole(role)

    return StreamingDeepAgent(name=name, role=role, **kwargs)


async def quick_chat(
    prompt: str, role: AgentRole = AgentRole.COORDINATOR, streaming: bool = True
) -> str:
    """
    Quick chat with a temporary agent.

    Args:
        prompt: The prompt to process
        role: Role of the temporary agent
        streaming: Enable streaming output

    Returns:
        str: The complete response
    """
    agent = StreamingDeepAgent(name="TempAgent", role=role, streaming=streaming)

    result = []
    async for token in agent.stream_response(prompt):
        result.append(token)
        if streaming and console:
            console.print(token, end="")

    return "".join(result)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Core classes
    "StreamingDeepAgent",
    "StreamingAgentOrchestrator",
    "StreamingCallback",
    # Enums and data models
    "AgentRole",
    "TaskStatus",
    "Task",
    # System prompts
    "SystemPrompts",
    # Utility functions
    "create_agent",
    "quick_chat",
    # Constants
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
]

# Entry point message
if __name__ == "__main__":
    print("Streaming Deep Agents Module")
    print("Use cli.py for interactive usage")

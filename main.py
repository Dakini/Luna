import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from core.llm_client.anthropic_llm_client import get_antropic_client
from core.tools.agent import Agent
from core.tools.workspace_manager import WorkspaceManager

load_dotenv()

MAX_OUTPUT_TOKENS_PER_TURN = 32768
MAX_TURNS = 200


def main():
    args = parse_arguments()

    if os.path.exists(args.logs_path):
        os.remove(args.logs_path)
    args.verbose = True

    logger_for_agent = logging.getLogger("agent_logs")
    logger_for_agent.setLevel(logging.DEBUG)
    logger_for_agent.addHandler(logging.FileHandler(args.logs_path))
    if args.verbose:
        logger_for_agent.addHandler(logging.StreamHandler())
    else:
        logger_for_agent.propagate = False

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        print("Please set the ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    if args.verbose:
        logger_for_agent.info(
            f"This tool allows you to interact with an agent, Press ctrl + c to exit or type exit or quit. "
        )

    client = get_antropic_client(model_name=args.model, use_caching=True)
    workspace_path = Path(args.workplace).resolve()

    workspace_manager = WorkspaceManager(
        root=workspace_path, container_workspace=args.use_container_workspace
    )

    agent = Agent(
        client=client,
        workspace_manager=workspace_manager,
        logger_for_agent_logs=logger_for_agent,
        max_output_tokens_per_turn=MAX_OUTPUT_TOKENS_PER_TURN,
        max_turns=MAX_TURNS,
        ask_for_permission=args.needs_permission,
        docker_container_id=args.docker_container_id,
    )

    history = InMemoryHistory()

    try:

        while True:
            user_input = prompt("You: ", history=history)
            history.append_string(user_input)
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                logger_for_agent.info("Exiting...")
                break

            logger_for_agent.info("\nAgent is thinking...")
            try:
                result = agent.run_agent(user_input, resume=True)
                logger_for_agent.info(f"Agent Result:\n{result}\n")

            except Exception as e:

                logger_for_agent.error(f"Error: {e}")
            logger_for_agent.info("\n" + "-" * 50 + "\n")

    except KeyboardInterrupt:
        logger_for_agent.info("Exiting...")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="A simple CLI tool to interact with an agent."
    )

    parser.add_argument(
        "--workplace",
        type=str,
        default=".",
        help="Path to the workspace directory (default: current directory).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Anthropic model to use.",
    )

    parser.add_argument(
        "--logs-path", type=str, default="agent_logs.txt", help="Path to save the logs"
    )

    parser.add_argument(
        "--needs-permission",
        "-p",
        help="Ask for permission before executing actions",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--use-container-workspace",
        help="(Optional) Path to the container workspace to run commands in.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--docker-container-id",
        help="(Optional) ID of the Docker container to run commands in.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        help="Minimise the output printed to stdout",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

from utils.client import Client
from utils.memory import save_history, load_history
from tools.weather_tool import get_current_weather
from tools.create_tools import create_tools

import psycopg
from psycopg.rows import dict_row

# import mlflow

# # mlflow.anthropic.autolog()
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("Anthropic")


# @mlflow.trace
def handle_chat(user_input, messages, client, session_id=None):

    messages.append({"role": "user", "content": user_input})
    messages = client.generate(messages)
    return messages


def main():
    # Create client
    tools = create_tools([get_current_weather])

    client = Client(tools=tools)
    messages = load_history()
    session_id = "session_1"
    tool_calls_found = True
    while tool_calls_found:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        messages = handle_chat(user_input, messages, client, session_id=session_id)
        save_history(messages)


if __name__ == "__main__":
    main()

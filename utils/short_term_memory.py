import json


def save_history(conversations, filename="memory.json"):
    with open(filename, "w") as f:
        json.dump(conversations, f, indent=2)


def load_history(filename="memory.json"):
    try:
        with open(filename, "r") as f:
            conversations = json.load(f)
    except FileNotFoundError:
        conversations = []
    return conversations


if __name__ == "__main__":
    messages = load_history()
    messages.append({"role": "user", "content": "Hello, how are you?"})
    save_history(messages)

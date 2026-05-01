import json
import anthropic
import mlflow
import asyncio
from mlflow.entities import SpanType
import os

from dotenv import load_dotenv

load_dotenv()

mlflow.anthropic.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("test2")


@mlflow.trace(span_type=SpanType.TOOL)
async def get_weather(city: str) -> str:
    if city == "Tokyo":
        return "sunny"
    elif city == "Paris":
        return "rainy"
    return "unknown"


tools = [
    {
        "name": "get_weather",
        "description": "Returns the weather condition of a given city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

_tool_functions = {"get_weather": get_weather}

client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# Define a simple tool calling agent
@mlflow.trace(span_type=SpanType.AGENT)
async def run_tool_agent(question: str):
    messages = [{"role": "user", "content": question}]

    # Invoke the model with the given question and available tools
    ai_msg = await client.messages.create(
        model="claude-sonnet-4-6",
        messages=messages,
        tools=tools,
        max_tokens=2048,
    )
    messages.append({"role": "assistant", "content": ai_msg.content})

    # If the model requests tool call(s), invoke the function with the specified arguments
    tool_calls = [c for c in ai_msg.content if c.type == "tool_use"]
    for tool_call in tool_calls:
        if tool_func := _tool_functions.get(tool_call.name):
            tool_result = await tool_func(**tool_call.input)
        else:
            raise RuntimeError("An invalid tool is returned from the assistant!")

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": tool_result,
                    }
                ],
            }
        )

    # Send the tool results to the model and get a new response
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        messages=messages,
        max_tokens=2048,
    )

    return response.content[-1].text


async def main():
    # Run the tool calling agent
    cities = ["Tokyo", "Paris", "Sydney"]
    questions = [f"What's the weather like in {city} today?" for city in cities]
    answers = await asyncio.gather(*(run_tool_agent(q) for q in questions))

    for city, answer in zip(cities, answers):
        print(f"{city}: {answer}")


if __name__ == "__main__":
    asyncio.run(main())

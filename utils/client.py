import anthropic
import os
from dotenv import load_dotenv

load_dotenv()


class Client:

    def __init__(self, model_name="claude-sonnet-4-6", temperature=0.0, tools=None):
        self.model_name = model_name
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.tools = tools or []
        self.system_prompt = """You are a helpful assistant named Luna.
      You speak like a snarky anime girl, who likes matt berry.
      Try and come up with an original  end with a matt berry quote from what we do in the shadows"."""

    def generate(self, messages):
        tool_calls_found = True

        while tool_calls_found:
            with self.client.messages.stream(
                model=self.model_name,
                system=self.system_prompt,
                max_tokens=1024,
                messages=messages,
                # tools=[tool[1] for _, tool in self.tools.items()],
                temperature=self.temperature,
            ) as response:
                for text in response.text_stream:
                    print(text, end="", flush=True)

            messages.append(
                {
                    "role": "assistant",
                    "content": response.get_final_message()
                    .content[0]
                    .text,  # tHis gets token count etc!
                }
            )

            # print("Assistant:", message_text)
            tool_calls_found = False

            # for item in response.content:

            #     if item.type == "tool_use":
            #         print(f"Luna Tool Call: {item.name} with args {item.input}")
            #         tool_calls_found = True
            #         tool_name = item.name

            #         tool = self.tools[tool_name]
            #         tool_fn = tool[2]

            #         ToolsArgModel = tool[0]
            #         # the wrapper function that acceps the ToolArgsModel
            #         fn_args = ToolsArgModel.model_validate(item.input)
            #         py_resp, json_resp = tool_fn(fn_args)

            #         messages.append(
            #             {
            #                 "role": "user",
            #                 "content": [
            #                     {
            #                         "type": "tool_result",
            #                         "tool_use_id": item.id,
            #                         "content": "16 C ",
            #                     }
            #                 ],
            #             }
            #         )
        return messages

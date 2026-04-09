import logging
from copy import deepcopy
from typing import Any, Optional

from core.llm_client import LLMClient
from core.prompts.system_prompt import SYSTEM_PROMPT
from core.tools.complete_tool import CompleteTool
from core.tools.message_classes import TextResult
from core.tools.weather import WeatherTool
from core.tools.workspace_manager import WorkspaceManager
from core.utils.dialog import DialogMessages
from core.utils.tool_common import LLMTool, ToolImplOutput


class Agent(LLMTool):
    name = "Luna Agent"
    decription = """A general agent that can accomplish tasks and answer questions.
    If you are faced with a task that involves more than a few steps, or if the task is complex or if the instructions are very long or vague
     try breaking down the task into smaller steps and call this tool multiple times """
    input_schema = {
        type: "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "The instruction to the agent",
            }
        },
        "required": ["instruction"],
    }

    def _get_system_prompt(self):
        """Get the system prompt, including any pending messages"""
        print("asd")
        return SYSTEM_PROMPT.format(workspace_root=self.workspace_manager.root)

    def __init__(
        self,
        client: LLMClient,
        workspace_manager: WorkspaceManager,  # Can update this later
        logger_for_agent_logs: logging.Logger = None,
        max_output_tokens_per_turn=8192,
        max_turns=10,
        use_prompt_budgeting: bool = True,
        ask_for_permission: bool = False,
        docker_container_id: Optional[str] = None,
    ):
        """Initialise the agent"""
        super().__init__()
        self.client = client
        self.workspace_manager = workspace_manager
        self.logger_for_agent_logs = logger_for_agent_logs
        self.max_output_tokens_per_turn = max_output_tokens_per_turn
        self.max_turns = max_turns

        self.use_prompt_budgeting = use_prompt_budgeting
        self.ask_for_permission = ask_for_permission
        self.docker_container_id = docker_container_id
        self.interupped = False

        self.dialog = DialogMessages(
            logger_for_agent_logs=self.logger_for_agent_logs,
            use_prompt_budgeting=self.use_prompt_budgeting,
        )

        self.complete_tool = CompleteTool()

        if docker_container_id is not None:
            # We will add some docker support soon!
            pass

        self.tools = [self.complete_tool]  # WE Will implement some soon!

        self.tools += [WeatherTool()]

    def run_impl(
        self, tool_input: dict[str, Any], dialog_messages: list[DialogMessages]
    ) -> ToolImplOutput:

        instruction = tool_input["instruction"]
        user_input_delimiter = f'{"-"*45} USER INPUT {"-"*45}\n{instruction}'
        self.logger_for_agent_logs.info(f"\n{user_input_delimiter}\n")

        self.dialog.add_user_message(instruction)
        self.interupped = False

        remaining_turns = self.max_turns
        while remaining_turns > 0:
            remaining_turns -= 1
            delimter = "-" * 45 + " NEW TURN " + "-" * 45
            self.logger_for_agent_logs.info(f"\n{delimter}\n")

            if self.dialog.use_prompt_budgeting:
                current_tok_amount = self.dialog.count_tokens()
                self.logger_for_agent_logs.info(
                    f"Current prompt token count: {current_tok_amount}"
                )
            tool_params = [tool.get_tool_params() for tool in self.tools]
            # check for duplicate tool names
            tool_names = [param.name for param in tool_params]
            sorted_names = sorted(tool_names)
            print("Hey")
            for i in range(len(sorted_names) - 1):
                if sorted_names[i] == sorted_names[i + 1]:
                    raise ValueError(f"Duplicate tool name found: {sorted_names[i]}")
            try:
                model_response, metadata = self.client.generate(
                    messages=self.dialog.get_messages_for_llm_client(),
                    max_tokens=self.max_output_tokens_per_turn,
                    tools=tool_params,
                    system_prompt=self._get_system_prompt(),
                )

                self.dialog.add_model_response(model_response)
                pending_tool_calls = self.dialog.get_pending_tool_calls()

                if len(pending_tool_calls) == 0:
                    self.logger_for_agent_logs.info("[no tools called]")
                    return ToolImplOutput(
                        tool_output=self.dialog.get_last_model_text_response(),
                        tool_result_message="Task Completed",
                    )
                if len(pending_tool_calls) > 1:
                    raise ValueError("Only one tool call per turn is supported")

                assert len(pending_tool_calls) == 1
                tool_call = pending_tool_calls[0]
                text_results = [
                    item for item in model_response if isinstance(item, TextResult)
                ]

                if len(text_results) > 0:
                    text_result = text_results[0]
                    self.logger_for_agent_logs.info(
                        f"Top level agent planning for next step: {text_result.text}"
                    )
                try:
                    tool = next(t for t in self.tools if t.name == tool_call.tool_name)
                except StopIteration as exc:
                    raise ValueError(
                        f"Tool with name {tool_call.tool_name} not found"
                    ) from exc

                try:
                    result = tool.run(tool_call.tool_input, deepcopy(self.dialog))
                    tool_input_str = "\n".join(
                        [f" - {k}: {v}" for k, v in tool_call.tool_input.items()]
                    )
                    log_message = f"Calling tool {tool_call.tool_name} with input:\n{tool_input_str}"
                    log_message += f"\nTool output: \n{result}\n"
                    self.logger_for_agent_logs.info(log_message)

                    if isinstance(result, tuple):
                        tool_result, _ = result
                    else:
                        tool_result = result
                    self.dialog.add_tool_call_result(tool_call, tool_result)

                    if self.complete_tool.should_stop:
                        ## Add a fake model response so that the next turn is the
                        # users turn in case they want to resume
                        self.dialog.add_model_response(
                            [TextResult(text="Completed the task")]
                        )
                        return ToolImplOutput(
                            tool_output=self.complete_tool.answer,
                            tool_result_message="Task Completed",
                        )
                except KeyboardInterrupt:
                    # Handle interuption during tool execution
                    self.interupped = True
                    interuppted_message = "Tool execution was interrupted by user"
                    self.dialog.add_tool_call_result(tool_call, interuppted_message)
                    self.dialog.add_model_response(
                        [
                            TextResult(
                                text="Tool execution interrupted by user. You can resume by providing new instructions for a new task. "
                            )
                        ]
                    )
                    return ToolImplOutput(
                        tool_output=interuppted_message,
                        tool_result_message=interuppted_message,
                    )

            except KeyboardInterrupt:
                self.interupped = True
                self.dialog.add_model_response(
                    [
                        TextResult(
                            text="Agent interrupted by user, You can resume by cproviding a new instruction"
                        )
                    ]
                )
                self.logger_for_agent_logs.error(f"Error during model generation: {e}")
                return ToolImplOutput(
                    tool_output="Agent Interrupted by user",
                    tool_result_message="Agent Interrupted by user",
                )

        agent_answer = "Agent did not complete the task after max turns"
        return ToolImplOutput(
            tool_output=agent_answer, tool_result_message=agent_answer
        )

    def get_tool_starting_message(self, tool_input: dict[str, Any]) -> str:
        return f"Agent started with instruction { tool_input["instruction"]}"

    def run_agent(
        self,
        instruction: str,
        resume: bool = False,
        orientation_instruction: str | None = None,
    ):
        """Start a new agent run

        Args:
            instruction: The instruciton to the agent.
            resume: Wherter to resume the agent from the previous state

        Returns:
            A Tuple(result, message)

        """
        if resume:
            assert self.dialog.is_user_turn()
        else:
            self.clear()
        tool_input = {"instruction": instruction}

        if orientation_instruction:
            tool_input["orientation_instruction"] = (
                orientation_instruction  # This is an optional instruction that is only shown in the first turn, and is meant to give the agent some context about the user or the task
            )
        return self.run(tool_input, self.dialog)

    def clear(self):
        self.dialog.clear()
        self.interupped = False

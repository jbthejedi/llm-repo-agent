from llm_repo_agent.prompts import system_prompt


def test_system_prompt_native_disallows_json_tool_calls():
    prompt = system_prompt("native")
    assert "Do NOT output a tool_call JSON in message content" in prompt
    assert "Output tool calls as a JSON object in assistant content" not in prompt


def test_system_prompt_json_allows_json_tool_calls():
    prompt = system_prompt("json")
    assert "Output tool calls as a JSON object in assistant content" in prompt
    assert "Do NOT output a tool_call JSON in message content" not in prompt

from llm_repo_agent.prompts import system_prompt


def test_system_prompt_native_disallows_json_tool_calls():
    prompt = system_prompt("native")
    assert "Do NOT output a tool_call JSON in message content" in prompt
    assert "Output tool calls as a JSON object in assistant content" not in prompt


def test_system_prompt_json_allows_json_tool_calls():
    prompt = system_prompt("json")
    assert "Output tool calls as a JSON object in assistant content" in prompt
    assert "Do NOT output a tool_call JSON in message content" not in prompt


def test_system_prompt_requires_root_list_files_first():
    prompt = system_prompt("json")
    assert "FIRST ACTION (MANDATORY)" in prompt
    assert "list_files with rel_dir='.'" in prompt
    assert "Do not call read_file/grep/write_file before this root listing" in prompt


def test_system_prompt_requires_list_files_required_args():
    prompt = system_prompt("json")
    assert "list_files must include both rel_dir and max_files" in prompt


def test_system_prompt_includes_write_rule():
    """Verify WRITE RULE is present to prevent hallucinated changes."""
    prompt = system_prompt("json")
    assert "WRITE RULE:" in prompt
    assert "MUST call write_file BEFORE using type='final'" in prompt
    assert "Listing changes in the final output does NOT make changes" in prompt


def test_system_prompt_native_also_includes_write_rule():
    """WRITE RULE should be in native prompts too."""
    prompt = system_prompt("native")
    assert "WRITE RULE:" in prompt
    assert "MUST call write_file BEFORE using type='final'" in prompt

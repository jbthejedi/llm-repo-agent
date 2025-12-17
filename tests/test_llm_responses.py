import pytest
from types import SimpleNamespace

from llm_repo_agent.llm import OpenAIResponsesLLM
from llm_repo_agent.actions import FinalAction


class DummyResp:
    def __init__(self, output_text):
        self.output = []
        self.output_text = output_text


def test_next_action_parses_first_json_and_warns():
    llm = OpenAIResponsesLLM()
    llm.client = SimpleNamespace(responses=SimpleNamespace(create=lambda *a, **k: DummyResp('{"type":"final","summary":"first"}{"type":"final","summary":"second"}')))

    with pytest.warns(UserWarning, match="Model returned extra JSON objects or trailing text"):
        action = llm.next_action([{"role": "system", "content": "hi"}])

    assert action.summary == "first"
    assert isinstance(action, FinalAction)


def test_next_action_raises_on_invalid_json():
    llm = OpenAIResponsesLLM()
    llm.client = SimpleNamespace(responses=SimpleNamespace(create=lambda *a, **k: DummyResp("not json at all")))

    with pytest.raises(RuntimeError, match="Could not parse model output as JSON"):
        llm.next_action([{"role": "system", "content": "hi"}])


def test_function_call_sets_last_raw():
    llm = OpenAIResponsesLLM()

    func_item = SimpleNamespace(type='function_call', name='list_files', arguments='{"rel_dir":".","max_files":5}')
    resp = SimpleNamespace(output=[func_item], output_text='')
    llm.client = SimpleNamespace(responses=SimpleNamespace(create=lambda *a, **k: resp))

    action = llm.next_action([{"role": "system", "content": "hi"}])
    assert action.name == 'list_files'
    assert hasattr(llm, '_last_raw')
    assert llm._last_raw['name'] == 'list_files'

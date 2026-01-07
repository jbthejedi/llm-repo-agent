# Migration Plan: Remove Legacy LLM Adaptors

## Status: ChatCompletionsLLM Implementation ✅ COMPLETE

The multi-turn `ChatCompletionsLLM` has been implemented and tested successfully:
- ✅ `openai-chat` provider passes quicksort eval
- ✅ Proper multi-turn conversation with `assistant`/`tool` roles
- ✅ Works with both OpenAI and Together AI
- ✅ All existing tests pass

## New Task: Migrate to ChatCompletionsLLM as Default

Now that multi-turn is proven to work better, migrate all providers to use `ChatCompletionsLLM`:

**Before:**
```
LLM Protocol
    ├── OpenAIResponsesLLM     (legacy Response API, keep for backward compat)
    ├── TogetherResponsesLLM   (broken single-turn, has empty args bug)
    └── ChatCompletionsLLM     (NEW multi-turn, works correctly)

Providers:
    openai → OpenAIResponsesLLM
    together → TogetherResponsesLLM
    openai-chat → ChatCompletionsLLM
    together-chat → ChatCompletionsLLM
```

**After:**
```
LLM Protocol
    └── ChatCompletionsLLM     (unified, multi-turn for all providers)

Providers:
    openai → ChatCompletionsLLM  (migrated)
    together → ChatCompletionsLLM  (migrated)
```

## Migration Strategy

**Goal:** Remove legacy LLM adaptors and make `ChatCompletionsLLM` the default for all providers.

### Step 1: Update Factory Builders (src/llm_repo_agent/llm.py)

**Change provider mappings:**

```python
# OLD (lines 525-543)
def _build_openai(cfg: LLMConfig) -> LLM:
    model = cfg.model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    return OpenAIResponsesLLM(...)  # Uses Response API

def _build_together(cfg: LLMConfig) -> LLM:
    model = cfg.model or os.getenv("TOGETHER_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo")
    return TogetherResponsesLLM(...)  # Broken single-turn

# NEW (already exists as _build_openai_chat, _build_together_chat)
```

**Action:** Repoint `openai` and `together` providers to use `ChatCompletionsLLM`:

```python
# Change _build_openai to use ChatCompletionsLLM
def _build_openai(cfg: LLMConfig) -> LLM:
    model = cfg.model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    return ChatCompletionsLLM(
        model=model,
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        base_url=None,  # OpenAI default
        api_key=None,
    )

# Change _build_together to use ChatCompletionsLLM
def _build_together(cfg: LLMConfig) -> LLM:
    model = cfg.model or os.getenv("TOGETHER_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo")
    base_url = cfg.together_base_url or os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
    return ChatCompletionsLLM(
        model=model,
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        base_url=base_url,
        api_key=cfg.together_api_key,
    )
```

**Remove duplicate registrations:**
```python
# DELETE these lines (redundant after migration):
LLMFactory.register("openai-chat", _build_openai_chat)
LLMFactory.register("together-chat", _build_together_chat)
```

### Step 2: Remove Legacy Classes (src/llm_repo_agent/llm.py)

**Delete these classes:**
1. `OpenAIResponsesLLM` (lines 34-147) - Response API, no longer needed
2. `TogetherResponsesLLM` (lines 150-272) - Broken single-turn implementation

**Keep:**
- `ChatCompletionsLLM` (lines 275-496) - The unified multi-turn implementation
- `LLMConfig` and `LLMFactory` - No changes needed

### Step 3: Remove Unused Tool Schema (src/llm_repo_agent/tool_schema.py)

**Delete:**
- `OPENAI_TOOLS` constant (only used by `OpenAIResponsesLLM`)
- `build_openai_tools()` function

**Keep:**
- `CHAT_COMPLETIONS_TOOLS` - Used by `ChatCompletionsLLM`
- All other functions

### Step 4: Update Tests

#### A. Update test_llm_factory.py

**Before:**
```python
from llm_repo_agent.llm import LLMFactory, LLMConfig, OpenAIResponsesLLM, TogetherResponsesLLM, ChatCompletionsLLM

def test_factory_builds_openai():
    llm = LLMFactory.build(LLMConfig(provider="openai"))
    assert isinstance(llm, OpenAIResponsesLLM)  # OLD

def test_factory_builds_together():
    llm = LLMFactory.build(LLMConfig(provider="together"))
    assert isinstance(llm, TogetherResponsesLLM)  # OLD
```

**After:**
```python
from llm_repo_agent.llm import LLMFactory, LLMConfig, ChatCompletionsLLM

def test_factory_builds_openai():
    llm = LLMFactory.build(LLMConfig(provider="openai"))
    assert isinstance(llm, ChatCompletionsLLM)  # NEW
    assert llm.base_url is None  # OpenAI default

def test_factory_builds_together():
    llm = LLMFactory.build(LLMConfig(provider="together"))
    assert isinstance(llm, ChatCompletionsLLM)  # NEW
    assert llm.base_url == "https://api.together.xyz/v1"
```

**Delete redundant tests:**
```python
# DELETE test_factory_builds_openai_chat() - redundant after migration
# DELETE test_factory_builds_together_chat() - redundant after migration
```

#### B. Delete or Rewrite test_llm_responses.py

**Option 1 - Delete entirely** (recommended if no unique coverage)
- Tests only covered `OpenAIResponsesLLM` and `TogetherResponsesLLM`
- Functionality is now tested via factory tests + agent integration tests

**Option 2 - Rewrite for ChatCompletionsLLM** (if we want explicit unit tests)
- Test `ChatCompletionsLLM.start_conversation()`
- Test `ChatCompletionsLLM.next_action()` with mocked responses
- Test multi-turn message building

**Recommendation:** Delete the file - integration tests cover this adequately.

### Step 5: Update Documentation

#### A. README.md (line 242)
**Before:**
```markdown
- llm.py — OpenAIResponsesLLM adapter
```

**After:**
```markdown
- llm.py — ChatCompletionsLLM adapter (unified multi-turn for all providers)
```

#### B. Blog Posts (blog_post/*.md)
Update references in:
- `blog_post/llm_repo_agent_blog_post_v0.md` (lines 95-97)
- `blog_post/llm_repo_agent_blog_post_v1.md` (lines 95-97)

**Before:**
```markdown
LLM adapter (`LLM`, `OpenAIResponsesLLM`) — turns prompts into typed Actions
OpenAIResponsesLLM sends the prompt, then parses the response into Actions
```

**After:**
```markdown
LLM adapter (`LLM`, `ChatCompletionsLLM`) — turns prompts into typed Actions
ChatCompletionsLLM maintains multi-turn conversation state with proper assistant/tool message roles
```

## Files to Modify (Summary)

1. **src/llm_repo_agent/llm.py**
   - Repoint `_build_openai()` to return `ChatCompletionsLLM`
   - Repoint `_build_together()` to return `ChatCompletionsLLM`
   - Delete `_build_openai_chat()` and `_build_together_chat()` functions
   - Delete `OpenAIResponsesLLM` class (lines 34-147)
   - Delete `TogetherResponsesLLM` class (lines 150-272)
   - Remove redundant factory registrations

2. **src/llm_repo_agent/tool_schema.py**
   - Delete `OPENAI_TOOLS` constant
   - Delete `build_openai_tools()` function

3. **tests/test_llm_factory.py**
   - Update `test_factory_builds_openai()` to expect `ChatCompletionsLLM`
   - Update `test_factory_builds_together()` to expect `ChatCompletionsLLM`
   - Delete `test_factory_builds_openai_chat()`
   - Delete `test_factory_builds_together_chat()`
   - Update imports

4. **tests/test_llm_responses.py**
   - DELETE entire file (3 tests for removed classes)

5. **README.md**
   - Update line 242 reference to ChatCompletionsLLM

6. **blog_post/llm_repo_agent_blog_post_v0.md**
   - Update lines 95-97

7. **blog_post/llm_repo_agent_blog_post_v1.md**
   - Update lines 95-97

## Testing Verification

After migration, run:

```bash
# All tests should pass
poetry run pytest tests/ -v

# Verify openai provider works
poetry run repo-agent eval --suite eval/suites/quicksort.json --llm-provider openai

# Verify together provider works
poetry run repo-agent eval --suite eval/suites/quicksort.json --llm-provider together --model "Qwen/Qwen2.5-7B-Instruct-Turbo"
```

## Benefits of This Migration

1. **Simplified codebase**: One LLM implementation instead of three
2. **Better performance**: Multi-turn conversation prevents empty args bug
3. **Unified interface**: Same behavior across OpenAI and Together AI
4. **Maintainability**: Single code path to test and debug
5. **Future-proof**: Chat Completions is the standard API format

## Risk Mitigation

**Low risk because:**
- ChatCompletionsLLM already tested and proven to work
- Agent.py already supports both multi-turn and single-turn interfaces
- All tests will be updated before merging
- Users can test before deploying

**Rollback plan (if needed):**
- Git revert the changes
- Legacy code is preserved in git history

---

## Last Edits

Also, change `agent.RepoAgent.run()` so that it doesn't use any code related to the legacy OpenAIResponsesLLM since we won't be using the Response API anymore. And add all the necessary test cases.
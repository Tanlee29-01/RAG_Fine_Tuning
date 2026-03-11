from src.generation.prompt_builder import build_prompt


def test_prompt_contains_question():
    prompt = build_prompt("What?", "Ctx")
    assert "What?" in prompt

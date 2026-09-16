import ast
from pathlib import Path


def test_gemma_campaign_keeps_labels_out_of_the_prompt_and_uses_temperature_zero() -> None:
    path = Path(__file__).parents[1] / "scripts" / "run_rome_gemma4_experiment.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT" for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Constant)
    assert isinstance(assignment.value.value, str)
    system_prompt = assignment.value.value
    assert "reference" not in system_prompt.lower()
    assert "similarite" not in system_prompt.lower()
    assert '"temperature": 0' in source
    assert '"response_format": {"type": "json_object"}' in source

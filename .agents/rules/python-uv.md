# Python Environment Rules

## 1. Virtual Environments with `uv`
- Never install dependencies into the system/base Python environment.
- Always manage environments with `uv` (`uv venv`, `uv add`, `uv pip install`, etc.).

## 2. Python Version Selection
- If a project already has a `uv` environment/config initialized, assume it uses the correct Python version.
- When creating a new `uv` environment from scratch, first check installed versions with `uv python list --only-installed` and select the highest installed stable version instead of defaulting to `python`/`python3`.

## 3. Execution
- Inside a `uv` project or environment, execute scripts with `uv run python ...` and commands with `uv run <command>`.

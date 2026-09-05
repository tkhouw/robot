# File Editing Tools Rule

- Always use the native `replace_file_content` tool when modifying existing files.
- Do NOT use shell commands (such as `cat`, `sed`, `echo`, or Python one-liners) to edit existing files.
- This ensures the user can clearly inspect and approve changes through visual diffs in the UI.

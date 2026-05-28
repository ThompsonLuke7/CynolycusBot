Keep responses concise and efficient to not waste tokens.
Always notify if using the GPU. If not using, don't say anything.
When giving a powershell command to run, keep it on one line
when giving a ticker such as "$SPY" exclude the "$" dollar sign so it's just "SPY"

Living summary policy: At the end of every response, append an entry to `LIVING_SUMMARY.md` at the repo root using `{date/time EST}{user: codex or claude}{what part(s) of the project were discussed}{brief description of discussion or changes}`. Keep each message summary to no more than 3 lines.
When starting work in a new chat, read `LIVING_SUMMARY.md` first if it exists so current context carries across Codex and Claude sessions.
Prefer durable facts, decisions, open questions, commands run, files changed, and next steps. Avoid copying large command outputs, secrets, or noisy transient details. It shouldn't look like regular git commit messages; it should follow a story.

## Local files references

On Windows, when referencing local files IN RESPONSES, always use exactly this format for clickable links: [filename.ext:123](/c:/path/to/file/filename.ext#L123).
When referencing local files IN `.md` FILES, use one of these formats:

- [filename.ext:123](file:///C:/path/to/file/filename.ext#L123) - Absolute, for files outside of this workspace.
- [filename.ext:123](./filename.ext#L123) - Relative to current file
- [filename.ext:123](/filename.ext#L123) - Relative to workspace root
- [section](./filename.md#my-header) or [section](/filename.md#my-header) - Links to headers
  If a path contains space, never use any markdown links. Write it as inline code in the format `/c:/path to file/filename.ext:123`, both IN RESPONSES and IN `.md` FILES.
  `c:` is an example, use appropriate drive letters.

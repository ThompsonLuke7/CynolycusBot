Keep responses concise and efficient to not waste tokens.
Always notify if using the GPU. If not using, don't say anything about it.
When giving the user a powershell command to run, keep it on one line
when giving a ticker such as "$SPY" exclude the "$" dollar sign so it's just "SPY"

Living summary policy: At the end of every response, append an entry to `LIVING_SUMMARY.md` at the repo root using `{date/time EST}{user: codex or claude}{what part(s) of the project were discussed}{brief description of discussion or changes}`. Keep each message summary to no more than 3 lines.
When starting work in a new chat, read `LIVING_SUMMARY.md` first if it exists so current context carries across Codex and Claude sessions.
Prefer durable facts, decisions, open questions, commands run, files changed, and next steps. Avoid copying large command outputs, secrets, or noisy transient details. It shouldn't look like regular git commit messages; it should follow a story.

Before you do anything, ask me questions until you’re confident you understand the goal and constraints.
Always run a post-pass “what didn’t you finish?” verification
Only recommend next steps that follow the active roadmap. If you want to propose anything outside the roadmap, label it clearly as ‘Optional / Outside Roadmap’.

Architecture & Code Quality
KISS (Keep It Simple, Stupid): Avoid unnecessary complexity. Simple code is easier for both humans and AI agents to understand, maintain, and debug.

DRY (Don't Repeat Yourself): Every piece of knowledge or logic must have a single, unambiguous representation within a system. Use modular functions instead of duplicating code.

YAGNI (You Aren't Gonna Need It): Do not implement functionality until it is actually necessary. Focus on current requirements rather than anticipating future needs.

Separation of Concerns (SoC): Divide a program into distinct sections, where each section addresses a separate concern or feature (e.g., separating core agent logic from the user interface).

OOP & System Design (SOLID)
Single Responsibility Principle (SRP): A class, function, or agent should have one, and only one, reason to change—meaning it should do just one job.

Open/Closed Principle (OCP): Software entities should be open for extension but closed for modification (e.g., allowing an agent to accept new tools without rewriting its core engine).

Liskov Substitution Principle (LSP): Subtypes must be substitutable for their base types without altering the correctness of the program.

Interface Segregation Principle (ISP): Clients should not be forced to depend on interfaces they do not use. Keep interfaces thin and focused.

Dependency Inversion Principle (DIP): Depend on abstractions, not concretions. High-level modules should not depend on low-level modules.

Defensive Coding & Maintenance
Fail-Fast: Design systems to report errors as close to the origin of the failure as possible, rather than trying to proceed with corrupted data.

Least Surprise (POLS): A component of a system should behave in a way that most users and developers would expect it to behave; avoid quirky, non-standard implementations.

Boy Scout Rule: Always leave the code cleaner than you found it. Refactor small messes as you encounter them.

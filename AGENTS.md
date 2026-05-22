Verification policy: Never run test suites, training, eval, or any command expected to take >10min unless explicitly requested by the user or within reason. If you foresee a task taking longer than 10 minutes to run, please try to break the runs into smaller chunks. If not possible, ask for permission to run the long-running task. Prefer static checks.

When giving a powershell command to run, keep it on one line
Do not run any ai training modules that would use the gpu.
when giving a ticker such as "$SPY" exclude the "$" dollar sign please so it would just be "SPY"

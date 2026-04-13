Verification policy: Never run test suites, training, eval, or any command expected to take >4min unless explicitly requested by the user. Prefer static checks only.

When giving a powershell command to run, keep it on one line
Do not run any ai training modules that would use the gpu.
when giving a ticker such as "$SPY" exclude the "$" dollar sign please so it would just be "SPY"
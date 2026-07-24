import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_live_server_dealer_ranker_flags_are_supported_by_combined_server() -> None:
    launcher = (REPO_ROOT / "scripts/run_live_server.sh").read_text()
    combined = (REPO_ROOT / "UI/combined_server.py").read_text()
    launcher_flags = set(re.findall(r'"(--dealer-ranker-[a-z-]+)"', launcher))
    parser_flags = set(
        re.findall(r'parser\.add_argument\(\s*"(--dealer-ranker-[a-z-]+)"', combined)
    )

    assert launcher_flags <= parser_flags, (
        "unsupported Dealer Ranker launcher flags: "
        + ", ".join(sorted(launcher_flags - parser_flags))
    )

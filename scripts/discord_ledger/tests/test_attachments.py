from __future__ import annotations

from bs4 import BeautifulSoup

from scripts.discord_ledger.attachments import attachment_links

CDN = "https://cdn.discordapp.com/attachments/1559060301955989554"


def message(body: str):
    html = f"<div class=chatlog__message-container data-message-id=1><div class=chatlog__message>{body}</div></div>"
    return BeautifulSoup(html, "html.parser").select_one("[data-message-id]")


def test_direct_and_forwarded_attachments_are_both_inventoried() -> None:
    msg = message(
        f'<div class=chatlog__attachment><a href="{CDN}/11/direct.png?ex=1"><img src="{CDN}/11/direct.png?ex=1" title="Image: direct.png (1 KB)"></a></div>'
        '<div class=chatlog__forwarded><div class=chatlog__forwarded-attachments>'
        f'<a href="{CDN}/22/forwarded.jpg?ex=1"><img class=chatlog__forwarded-attachment src="{CDN}/22/forwarded.jpg?ex=1"></a>'
        '</div></div>'
    )
    links = list(attachment_links(msg))
    assert [name for _, name, _ in links] == ["direct.png", "forwarded.jpg"]


def test_non_discord_forwarded_links_are_rejected() -> None:
    msg = message('<div class=chatlog__forwarded-attachments><a href="https://example.com/x.png">x</a></div>')
    assert list(attachment_links(msg)) == []

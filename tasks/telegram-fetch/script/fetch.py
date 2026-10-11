#!/usr/bin/env python3
"""Fetch a public Telegram channel preview and store posts as Markdown."""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_URL = "https://t.me/s/vip115hot"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "output" / "result.md"
USER_AGENT = "Mozilla/5.0 (compatible; TelegramMarkdownFetcher/1.0)"


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["Node | str"] = field(default_factory=list)

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())


class Parser(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, {key: value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


@dataclass(frozen=True)
class Post:
    post_id: int
    channel: str
    date: str
    url: str
    body: str

    @property
    def fingerprint(self) -> str:
        # Ignore spacing differences so repeated/forwarded text is still deduplicated.
        canonical = re.sub(r"\s+", " ", self.body).strip().casefold()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def walk(node: Node) -> Iterator[Node]:
    yield node
    for child in node.children:
        if isinstance(child, Node):
            yield from walk(child)


def find_class(node: Node, name: str) -> Node | None:
    return next((item for item in walk(node) if name in item.classes), None)


def to_markdown(node: Node | str) -> str:
    if isinstance(node, str):
        return node
    if node.tag == "br":
        return "\n"
    value = "".join(to_markdown(child) for child in node.children)
    if node.tag == "a":
        label = value.strip()
        href = node.attrs.get("href", "")
        if not href or href.startswith("?"):
            return label
        href = urljoin("https://t.me/", href)
        return href if label == href else f"[{label}]({href})"
    if node.tag in {"b", "strong"} and value.strip():
        return f"**{value.strip()}**"
    if node.tag in {"i", "em"} and value.strip() and "emoji" not in node.classes:
        return f"*{value.strip()}*"
    return value


def normalize(value: str) -> str:
    value = html.unescape(value).replace("\xa0", " ")
    value = re.sub(r"[ \t]+\n", "\n", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def parse_page(source: str, base_url: str) -> tuple[list[Post], str | None, str]:
    parser = Parser()
    parser.feed(source)
    title_tag = next((n for n in walk(parser.root) if n.tag == "meta" and n.attrs.get("property") == "og:title"), None)
    title = title_tag.attrs.get("content", "Telegram channel") if title_tag else "Telegram channel"
    posts: list[Post] = []
    for node in walk(parser.root):
        identity = node.attrs.get("data-post", "")
        if "js-widget_message" not in node.classes or not re.fullmatch(r"[^/]+/\d+", identity):
            continue
        channel, raw_id = identity.split("/", 1)
        body_tag = find_class(node, "js-message_text")
        if body_tag is None:
            continue
        time_tag = next((n for n in walk(node) if n.tag == "time" and n.attrs.get("datetime")), None)
        posts.append(Post(
            int(raw_id), channel,
            time_tag.attrs.get("datetime", "") if time_tag else "",
            f"https://t.me/{channel}/{raw_id}",
            normalize(to_markdown(body_tag)),
        ))
    more = find_class(parser.root, "js-messages_more")
    previous = urljoin(base_url, more.attrs["href"]) if more and more.attrs.get("href") else None
    return posts, previous, title


def fetch(url: str, timeout: float, retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"})
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"请求失败：{url} ({last_error})")


BLOCK_RE = re.compile(
    r"<!-- telegram-post: (?P<channel>[^/]+)/(?P<id>\d+); sha256: [0-9a-f]{64} -->\n"
    r"## .*?\n\n- 时间：(?P<date>.*?)\n- 原文：\[Telegram \d+\]\((?P<url>.*?)\)\n\n"
    r"(?P<body>.*?)(?:\n\n---\n|\Z)", re.S,
)


def load_existing(path: Path) -> dict[int, Post]:
    if not path.exists():
        return {}
    result: dict[int, Post] = {}
    for match in BLOCK_RE.finditer(path.read_text(encoding="utf-8")):
        post_id = int(match.group("id"))
        result[post_id] = Post(post_id, match.group("channel"), match.group("date"), match.group("url"), match.group("body").strip())
    return result


def deduplicate(posts: dict[int, Post]) -> dict[int, Post]:
    """Keep the newest post for duplicate IDs and duplicate normalized bodies."""
    seen_hashes: set[str] = set()
    unique: dict[int, Post] = {}
    for post in sorted(posts.values(), key=lambda item: item.post_id, reverse=True):
        if post.fingerprint in seen_hashes:
            continue
        seen_hashes.add(post.fingerprint)
        unique[post.post_id] = post
    return unique


def heading(post: Post) -> str:
    value = re.sub(r"[*_`\[\]]", "", post.body.splitlines()[0] if post.body else "")
    return re.sub(r"^名称[：:]\s*", "", value).strip()[:100] or f"消息 {post.post_id}"


def write_result(path: Path, title: str, source_url: str, posts: dict[int, Post]) -> None:
    lines = [
        f"# {title}", "", f"- 来源：[{source_url}]({source_url})",
        f"- 更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 去重后消息数：{len(posts)}", "",
    ]
    for post in sorted(posts.values(), key=lambda item: item.post_id, reverse=True):
        lines.extend([
            f"<!-- telegram-post: {post.channel}/{post.post_id}; sha256: {post.fingerprint} -->",
            f"## {heading(post)}", "", f"- 时间：{post.date}",
            f"- 原文：[Telegram {post.post_id}]({post.url})", "", post.body, "", "---", "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取 Telegram 公开频道并生成去重 Markdown")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--pages", type=int, default=1, help="最多抓取页数，默认 1")
    ap.add_argument("--all", action="store_true", help="抓取全部可访问历史页")
    ap.add_argument("--fresh", action="store_true", help="不读取已有结果")
    ap.add_argument("--delay", type=float, default=0.8)
    ap.add_argument("--timeout", type=float, default=20)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()
    if args.pages < 1 or args.delay < 0 or args.timeout <= 0 or args.retries < 0:
        ap.error("参数值超出有效范围")
    parsed = urlparse(args.url)
    if parsed.scheme != "https" or parsed.netloc not in {"t.me", "telegram.me"}:
        ap.error("--url 必须是 Telegram HTTPS 公开预览地址")

    output = args.output.expanduser().resolve()
    posts = {} if args.fresh else load_existing(output)
    old_ids = set(posts)
    page_url: str | None = args.url
    title = "Telegram channel"
    page_number = 0
    try:
        while page_url and (args.all or page_number < args.pages):
            page_posts, previous, title = parse_page(fetch(page_url, args.timeout, args.retries), args.url)
            page_number += 1
            for post in page_posts:  # Same ID updates replace older content.
                posts[post.post_id] = post
            print(f"第 {page_number} 页：读取 {len(page_posts)} 条", file=sys.stderr)
            if old_ids and any(post.post_id in old_ids for post in page_posts):
                break
            page_url = previous
            if page_url and (args.all or page_number < args.pages):
                time.sleep(args.delay)
    except (RuntimeError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    before = len(posts)
    posts = deduplicate(posts)
    write_result(output, title, args.url, posts)
    print(f"已写入 {output}：{len(posts)} 条（内容去重 {before - len(posts)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

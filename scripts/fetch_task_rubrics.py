"""Fetch public simulation scoring tables; print a reviewable JSON snapshot.

Run with Python 3 and redirect stdout only when intentionally refreshing the
reviewed snapshot. No simulator implementation or live evaluation is accessed.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import re
import urllib.request

BASE = "https://robodojo-benchmark.com"


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"th", "td"}:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def fetch(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return response.read().decode("utf-8")


def parse_scoring(html):
    section = html.split('id="scoring"', 1)[1]
    table = section[section.index("<table"):section.index("</table>") + 8]
    parser = TableParser()
    parser.feed(table)
    if parser.rows[0] != ["Score", "Condition"] or len(parser.rows) < 2:
        raise ValueError("Unexpected wiki scoring table")
    if any(len(row) != 2 or not all(row) for row in parser.rows[1:]):
        raise ValueError("Incomplete scoring rows")
    return [{"points": points, "condition": condition} for points, condition in parser.rows[1:]]


def main():
    index = fetch(BASE + "/doc/sim-tasks/make-kong/")
    paths = sorted(set(re.findall(r'href="(/doc/sim-tasks/[a-z0-9-]+/)"', index)))
    paths = [path for path in paths if path.rsplit("/", 2)[1] not in
             {"domain-randomization", "parallel-environments", "dlc"}]

    def task(path):
        url = BASE + path
        return path.rsplit("/", 2)[1].replace("-", "_"), {
            "source_url": url, "criteria": parse_scoring(fetch(url))}

    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = dict(pool.map(task, paths))
    print(json.dumps({"schema_version": 1,
                      "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                      "tasks": tasks}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

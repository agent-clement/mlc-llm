#!/usr/bin/env python3
"""Summarize CUDA graph captures in an MLC Relax debug dump."""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path
from typing import Any


DEF_RE = re.compile(r"^\s+def\s+([A-Za-z0-9_]+)\(")
CAPTURE_RE = re.compile(r"^(?P<parent>.+)_cuda_graph_capture(?P<index>\d*)$")
CALL_TIR_RE = re.compile(r"cls\.([A-Za-z0-9_]+),")
LOWERED_TIR_RE = re.compile(r"\bcls\.([A-Za-z0-9_]+)\(")
PACKED_RE = re.compile(r'R\.call_(?:dps_)?(?:pure_)?packed\("([^"]+)"')
BUILTIN_RE = re.compile(r'R\.call_builtin_with_ctx\("([^"]+)"')


def _is_top_level_def(line: str) -> bool:
    return line.startswith("    def ")


def capture_parent_name(name: str) -> str | None:
    match = CAPTURE_RE.match(name)
    if match is None:
        return None
    return match.group("parent")


def capture_index(name: str) -> int | None:
    match = CAPTURE_RE.match(name)
    if match is None:
        return None
    index = match.group("index")
    if not index:
        return 0
    return int(index)


def parse_functions(path: Path) -> dict[str, dict[str, object]]:
    funcs: dict[str, dict[str, object]] = {}
    current: str | None = None

    with path.open("r", encoding="utf-8") as file:
        for lineno, line in enumerate(file, 1):
            match = DEF_RE.match(line)
            if match and _is_top_level_def(line):
                current = match.group(1)
                funcs[current] = {
                    "lineno": lineno,
                    "lines": 1,
                    "tir": [],
                    "packed": [],
                    "builtin": [],
                    "events": [],
                }
                continue

            if current is None:
                continue

            if _is_top_level_def(line):
                current = None
                continue

            item = funcs[current]
            item["lines"] = int(item["lines"]) + 1

            if "R.call_tir" in line:
                tir_match = CALL_TIR_RE.search(line)
                if tir_match:
                    name = tir_match.group(1)
                    item["tir"].append(name)
                    item["events"].append((lineno, "tir", name))
            for lowered_tir in LOWERED_TIR_RE.findall(line):
                if lowered_tir != "cuda_graph_alloc":
                    item["tir"].append(lowered_tir)
                    item["events"].append((lineno, "tir", lowered_tir))
            for packed in PACKED_RE.findall(line):
                item["packed"].append(packed)
                item["events"].append((lineno, "packed", packed))
            for builtin in BUILTIN_RE.findall(line):
                item["builtin"].append(builtin)
                item["events"].append((lineno, "builtin", builtin))

    return funcs


def analyze(path: Path, function_filter: str | None, parent_filter: str | None) -> None:
    funcs = parse_functions(path)

    captures = {}
    for name, item in funcs.items():
        parent = capture_parent_name(name)
        if parent is None:
            continue
        if function_filter is not None and function_filter not in name:
            continue
        if parent_filter is not None and parent != parent_filter:
            continue
        captures[name] = item

    print(f"file={path}")
    print(f"functions={len(funcs)} captures={len(captures)}")

    by_prefix: collections.Counter[str] = collections.Counter()
    for name in captures:
        parent = capture_parent_name(name)
        if parent is not None:
            by_prefix[parent] += 1
    print("captures_by_parent=" + ", ".join(f"{k}:{v}" for k, v in sorted(by_prefix.items())))

    tir_counter: collections.Counter[str] = collections.Counter()
    packed_counter: collections.Counter[str] = collections.Counter()
    builtin_counter: collections.Counter[str] = collections.Counter()
    for item in captures.values():
        tir_counter.update(item["tir"])
        packed_counter.update(item["packed"])
        builtin_counter.update(item["builtin"])

    def show_counter(title: str, counter: collections.Counter[str], limit: int) -> None:
        print(title)
        for name, count in counter.most_common(limit):
            print(f"  {count:4d} {name}")

    show_counter("top_tir_calls", tir_counter, 30)
    show_counter("packed_calls", packed_counter, 30)
    show_counter("builtin_calls", builtin_counter, 30)

    print("largest_captures")
    ordered = sorted(
        captures.items(),
        key=lambda kv: (int(kv[1]["lines"]), len(kv[1]["tir"])),
        reverse=True,
    )
    for name, item in ordered[:20]:
        tir = collections.Counter(item["tir"])
        packed = collections.Counter(item["packed"])
        print(
            f"  {name}:{item['lineno']} lines={item['lines']} "
            f"tir={len(item['tir'])} packed={len(item['packed'])} "
            f"builtin={len(item['builtin'])}"
        )
        if tir:
            print("    tir=" + ", ".join(f"{k}:{v}" for k, v in tir.most_common(8)))
        if packed:
            print("    packed=" + ", ".join(f"{k}:{v}" for k, v in packed.most_common(8)))


def get_capture_region_summaries(
    path: Path,
    function_filter: str | None = None,
    parent_filter: str | None = None,
    max_events: int = 6,
) -> list[dict[str, Any]]:
    funcs = parse_functions(path)
    regions: list[dict[str, Any]] = []

    for name, item in funcs.items():
        parent = capture_parent_name(name)
        index = capture_index(name)
        if parent is None or index is None:
            continue
        if function_filter is not None and function_filter not in name:
            continue
        if parent_filter is not None and parent != parent_filter:
            continue

        events = list(item["events"])
        regions.append(
            {
                "name": name,
                "parent": parent,
                "index": index,
                "lineno": item["lineno"],
                "lines": item["lines"],
                "tir_count": len(item["tir"]),
                "packed_count": len(item["packed"]),
                "builtin_count": len(item["builtin"]),
                "tir": collections.Counter(item["tir"]),
                "packed": collections.Counter(item["packed"]),
                "builtin": collections.Counter(item["builtin"]),
                "first_events": events[:max_events],
                "last_events": events[-max_events:] if max_events else [],
            }
        )

    return sorted(regions, key=lambda item: (item["parent"], item["index"]))


def _format_events(events: list[tuple[int, str, str]]) -> str:
    if not events:
        return "<none>"
    return "; ".join(f"{lineno}:{kind}:{name}" for lineno, kind, name in events)


def print_capture_regions(
    path: Path,
    function_filter: str | None,
    parent_filter: str | None,
    max_events: int,
) -> None:
    regions = get_capture_region_summaries(
        path,
        function_filter=function_filter,
        parent_filter=parent_filter,
        max_events=max_events,
    )
    print(f"file={path}")
    print(f"capture_regions={len(regions)}")
    for region in regions:
        print(
            f"  {region['name']}:{region['lineno']} "
            f"parent={region['parent']} index={region['index']} "
            f"lines={region['lines']} tir={region['tir_count']} "
            f"packed={region['packed_count']} builtin={region['builtin_count']}"
        )
        if region["packed"]:
            print(
                "    packed="
                + ", ".join(f"{k}:{v}" for k, v in region["packed"].most_common(8))
            )
        if region["builtin"]:
            print(
                "    builtin="
                + ", ".join(f"{k}:{v}" for k, v in region["builtin"].most_common(8))
            )
        print(f"    first={_format_events(region['first_events'])}")
        print(f"    last={_format_events(region['last_events'])}")


def inspect_function_stats(path: Path, function_name: str) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for index, line in enumerate(lines):
        match = DEF_RE.match(line)
        if match and _is_top_level_def(line) and match.group(1) == function_name:
            start = index
            break
    if start is None:
        raise SystemExit(f"Function not found: {function_name}")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = DEF_RE.match(lines[index])
        if match and _is_top_level_def(lines[index]):
            end = index
            break

    body = lines[start:end]
    call_sites = []
    capture_counter: collections.Counter[str] = collections.Counter()
    packed_counter: collections.Counter[str] = collections.Counter()
    builtin_counter: collections.Counter[str] = collections.Counter()
    tir_counter: collections.Counter[str] = collections.Counter()
    for offset, line in enumerate(body, start + 1):
        if "vm.builtin.cuda_graph.run_or_capture" in line:
            capture_counter[function_name] += 1
            call_sites.append((offset, "cuda_graph", line.strip()))
            continue
        for packed in PACKED_RE.findall(line):
            packed_counter[packed] += 1
            call_sites.append((offset, "packed", f"{packed}: {line.strip()}"))
        for builtin in BUILTIN_RE.findall(line):
            builtin_counter[builtin] += 1
            call_sites.append((offset, "builtin", f"{builtin}: {line.strip()}"))
        if "R.call_tir" in line:
            tir_match = CALL_TIR_RE.search(line)
            if tir_match:
                tir_counter[tir_match.group(1)] += 1

    return {
        "path": path,
        "function_name": function_name,
        "lineno": start + 1,
        "lines": len(body),
        "cuda_graph_calls": sum(capture_counter.values()),
        "packed": packed_counter,
        "builtin": builtin_counter,
        "tir": tir_counter,
        "call_sites": call_sites,
    }


def inspect_function(
    path: Path,
    function_name: str,
    *,
    expect_cuda_graph_calls: int | None = None,
    expect_packed_call: list[str] | None = None,
    expect_no_packed_call: list[str] | None = None,
    max_call_sites: int = 200,
) -> None:
    stats = inspect_function_stats(path, function_name)
    print(f"file={path}")
    print(
        f"function={function_name} lineno={stats['lineno']} "
        f"lines={stats['lines']}"
    )
    print(f"cuda_graph_calls={stats['cuda_graph_calls']}")
    print("packed_calls")
    for name, count in stats["packed"].most_common(50):
        print(f"  {count:4d} {name}")
    print("builtin_calls")
    for name, count in stats["builtin"].most_common(50):
        print(f"  {count:4d} {name}")
    print("tir_calls")
    for name, count in stats["tir"].most_common(30):
        print(f"  {count:4d} {name}")
    print("call_sites")
    for lineno, kind, text in stats["call_sites"][:max_call_sites]:
        print(f"  {lineno}: {kind}: {text}")
    check_function_expectations(
        stats,
        expect_cuda_graph_calls=expect_cuda_graph_calls,
        expect_packed_call=expect_packed_call or [],
        expect_no_packed_call=expect_no_packed_call or [],
    )


def check_function_expectations(
    stats: dict[str, Any],
    *,
    expect_cuda_graph_calls: int | None,
    expect_packed_call: list[str],
    expect_no_packed_call: list[str],
) -> None:
    if (
        expect_cuda_graph_calls is not None
        and stats["cuda_graph_calls"] != expect_cuda_graph_calls
    ):
        raise SystemExit(
            "Expected "
            f"{expect_cuda_graph_calls} CUDA graph calls in {stats['function_name']}, "
            f"found {stats['cuda_graph_calls']}"
        )

    for spec in expect_packed_call:
        if "=" in spec:
            name, expected_text = spec.rsplit("=", 1)
            expected = int(expected_text)
        else:
            name = spec
            expected = 1
        actual = stats["packed"].get(name, 0)
        if actual != expected:
            raise SystemExit(f"Expected packed call {name}={expected}, found {actual}")

    for name in expect_no_packed_call:
        actual = stats["packed"].get(name, 0)
        if actual:
            raise SystemExit(f"Expected no packed call {name}, found {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, help="Path to debug-phase6.py")
    parser.add_argument("--filter", help="Only include capture functions whose name contains this text")
    parser.add_argument("--parent", help="Only include capture functions lifted from this parent")
    parser.add_argument("--function", help="Inspect non-captured calls inside this top-level function")
    parser.add_argument(
        "--regions",
        action="store_true",
        help="Print ordered capture-region summaries instead of aggregate counters",
    )
    parser.add_argument("--expect-cuda-graph-calls", type=int)
    parser.add_argument(
        "--expect-packed-call",
        action="append",
        default=[],
        metavar="SYMBOL[=COUNT]",
        help="Require a packed call count when used with --function",
    )
    parser.add_argument(
        "--expect-no-packed-call",
        action="append",
        default=[],
        metavar="SYMBOL",
        help="Require a packed call to be absent when used with --function",
    )
    parser.add_argument("--max-call-sites", type=int, default=200)
    parser.add_argument("--max-region-events", type=int, default=6)
    args = parser.parse_args()
    if args.function:
        inspect_function(
            args.dump,
            args.function,
            expect_cuda_graph_calls=args.expect_cuda_graph_calls,
            expect_packed_call=args.expect_packed_call,
            expect_no_packed_call=args.expect_no_packed_call,
            max_call_sites=args.max_call_sites,
        )
    elif args.regions:
        print_capture_regions(
            args.dump,
            args.filter,
            args.parent,
            max_events=args.max_region_events,
        )
    else:
        analyze(args.dump, args.filter, args.parent)


if __name__ == "__main__":
    main()

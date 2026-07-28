"""Count what RENDERS, not what calls t().

The previous check measured "keys inside t() that have a translation" and
reported 377/377, 0 missing — while the English homepage still showed Chinese,
because module-level const tables were rendered directly and never reached t()
at all. A measurement that can only see inside t() cannot detect a string that
never got there: it is the same circular mistake as a discriminator that scores
whether the feature exists rather than whether it works.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "web"
HAN = re.compile("[一-鿿]")
TCALL = re.compile(r"\bt\(\s*(['\"])(?:(?!\1)[^\\]|\\.)*\1(?:\s*,\s*\{[^}]*\})?\s*\)")
# `{item.label}` etc: a bare member expression rendered straight into JSX
BARE_RENDER = re.compile(r"\{\s*(\w+)\.(label|title|desc|meta|text|subtitle|name)\s*\}")
COMMENT = re.compile(r"^\s*(//|\*|/\*)")


def _tsx_files():
    for directory in ("app", "components"):
        root = WEB / directory
        if root.exists():
            yield from sorted(root.rglob("*.tsx"))


def _module_const_chinese_fields(src: str) -> set[str]:
    """Field names that a module-level const in THIS file fills with Chinese.

    Narrow on purpose. `{song.title}` is a real song's name — translating it
    would be a bug, not a fix. Only fields whose values are hardcoded Chinese in
    a const table above the first component are UI text.
    """
    head = src.split("export default function", 1)[0]
    fields = set()
    for m in re.finditer(r"(\w+)\s*:\s*'([^']*)'", head):
        if HAN.search(m.group(2)):
            fields.add(m.group(1))
    return fields


def test_no_module_constant_is_rendered_without_translation():
    """`{item.label}` straight into JSX renders whatever the const table holds.

    This is exactly how the homepage quick-example cards, the sidebar links and
    the settings tabs stayed Chinese in English mode while the old check
    reported "0 missing translations".
    """
    offenders = []
    for path in _tsx_files():
        src = path.read_text(encoding="utf-8")
        ui_fields = _module_const_chinese_fields(src)
        if not ui_fields:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            if COMMENT.match(line) or "key={" in line:
                continue
            for m in BARE_RENDER.finditer(line):
                obj, field = m.group(1), m.group(2)
                if field not in ui_fields:
                    continue                      # data, not a UI label
                if f"t({obj}.{field})" in line:
                    continue
                offenders.append(f"{path.relative_to(WEB)}:{i}  {m.group(0)}")
    assert not offenders, (
        "module-constant labels rendered without t() — these stay Chinese in "
        "English mode:\n" + "\n".join(offenders))


def test_every_translatable_key_has_an_entry():
    """Keys reaching t() must be translated, or English mode falls back."""
    dict_src = (WEB / "lib" / "i18n-en.ts").read_text(encoding="utf-8")
    # Values may be double-quoted — "Doesn't fit" has to be, because of the
    # apostrophe. A single-quote-only pattern reported that present key as
    # missing, which is the same measurement bug twice over: the check was
    # wrong, not the data.
    have = set(re.findall(r"'((?:[^'\\]|\\.)*)':\s*['\"]", dict_src))

    used, missing = set(), []
    for path in _tsx_files():
        for m in TCALL.finditer(path.read_text(encoding="utf-8")):
            key = m.group(0)
            inner = re.match(r"\bt\(\s*'((?:[^'\\]|\\.)*)'", key)
            if inner and HAN.search(inner.group(1)):
                used.add(inner.group(1))
    for key in sorted(used):
        if key not in have:
            missing.append(key)
    assert not missing, f"{len(missing)} keys fall back to Chinese: {missing[:10]}"


def test_html_lang_and_metadata_are_not_hardcoded_chinese():
    """A visitor from GitHub gets the English page; the tab title must match."""
    layout = (WEB / "app" / "layout.tsx").read_text(encoding="utf-8")
    assert 'lang="zh-CN"' not in layout, "<html lang> is pinned to Chinese"
    meta = re.search(r"export const metadata[^}]*\}", layout, re.S)
    assert meta and not HAN.search(meta.group(0)), "page metadata is still Chinese"


@pytest.mark.parametrize("path,needle", [
    ("components/Landing/ProductIntro.tsx", "t(example.title)"),
    ("components/Navigation/NavItem.tsx", "t(label)"),
    ("components/Settings/SettingsPanel.tsx", "t(tab.label)"),
    ("components/Content/SlateFeedback.tsx", "t(option.label)"),
])
def test_known_constant_tables_are_translated_at_the_render_site(path, needle):
    """Spot-checks for the tables the review actually caught in the browser."""
    assert needle in (WEB / path).read_text(encoding="utf-8"), f"{path} missing {needle}"

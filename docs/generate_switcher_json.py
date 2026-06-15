#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

PACKAGE_ROOT = '/acoular'
MIN_VERSION = 'v26.01'
OUTPUT = Path('_static/switcher.json')


def version_key(tag: str) -> tuple[int, int]:
    match = re.match(r'v(\d+)\.(\d+)$', tag)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)


def normalize_version(tag: str) -> str:
    return tag.removeprefix('v')


def iter_git_tags(repo_root: Path) -> list[str]:
    git_dir = repo_root / '.git'
    refs_dir = git_dir / 'refs' / 'tags'
    packed_refs = git_dir / 'packed-refs'
    tags: set[str] = set()

    if refs_dir.exists():
        tags.update(path.relative_to(refs_dir).as_posix() for path in refs_dir.rglob('*') if path.is_file())

    if packed_refs.exists():
        with packed_refs.open(encoding='utf-8') as file_handle:
            for line in file_handle:
                if line.startswith(('#', '^')):
                    continue
                parts = line.strip().split(' ', maxsplit=1)
                if len(parts) != 2:
                    continue
                ref_name = parts[1]
                if ref_name.startswith('refs/tags/'):
                    tags.add(ref_name.removeprefix('refs/tags/'))

    return sorted(tags)


repo_root = Path('..').resolve()
tags = iter_git_tags(repo_root)
filtered_tags = sorted(
    [tag for tag in tags if re.match(r'^v\d+\.\d+$', tag)],
    key=version_key,
    reverse=True,
)

versions: list[dict[str, str | bool]] = [
    {
        'name': 'dev',
        'version': 'dev',
        'url': f'{PACKAGE_ROOT}/dev/',
        'type': 'branch',
    }
]

latest_stable = None
for tag in filtered_tags:
    if version_key(tag) < version_key(MIN_VERSION):
        continue
    if latest_stable is None:
        latest_stable = tag
        versions.insert(
            0,
            {
                'name': 'stable',
                'version': normalize_version(tag),
                'url': f'{PACKAGE_ROOT}/',
                'type': 'tag',
                'preferred': True,
            },
        )
    versions.append(
        {
            'name': tag,
            'version': normalize_version(tag),
            'url': f'{PACKAGE_ROOT}/{tag}/',
            'type': 'tag',
        }
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open('w') as file_handle:
    json.dump(versions, file_handle, indent=2)

print(f'Generated {OUTPUT} with {len(versions)} versions')
if latest_stable:
    print(f'Latest stable release: {latest_stable}')

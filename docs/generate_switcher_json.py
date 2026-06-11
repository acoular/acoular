#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
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


result = subprocess.run(
    ['git', 'tag', '--merged', 'HEAD', '--list', 'v*'],
    capture_output=True,
    text=True,
    cwd='..',
    check=True,
)
tags = result.stdout.strip().split('\n') if result.stdout.strip() else []
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
                'name': f'{tag} (stable)',
                'version': normalize_version(tag),
                'url': f'{PACKAGE_ROOT}/',
                'type': 'tag',
                'preferred': True,
            },
        )
    else:
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

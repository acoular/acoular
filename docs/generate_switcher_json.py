#!/usr/bin/env python3
import json
import re
import subprocess

# Get all version tags starting with 'v', sorted newest first
result = subprocess.run(['git', 'tag', '--merged', 'HEAD', '--list', 'v*'], capture_output=True, text=True, cwd='..')
tags = result.stdout.strip().split('\n') if result.stdout.strip() else []


# Filter and sort tags
def version_key(tag):
    # Extract version numbers for sorting
    match = re.match(r'v(\d+)\.(\d+)', tag)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)


filtered_tags = [tag for tag in tags if re.match(r'^v\d+\.\d+$', tag)]
filtered_tags.sort(key=version_key, reverse=True)

MIN_VERSION = 'v25.01'

# Build JSON structure
versions = [{'name': 'latest', 'url': '/en/latest/', 'type': 'branch'}]

latest_stable = None
for tag in filtered_tags:
    if version_key(tag) >= version_key(MIN_VERSION):  # semantic comparison, not lexicographic
        if latest_stable is None:
            # First matching tag is the latest stable release
            latest_stable = tag
            versions.append({'name': f'{tag} (stable)', 'url': f'/en/{tag}/', 'type': 'tag', 'preferred': True})
        else:
            versions.append({'name': tag, 'url': f'/en/{tag}/', 'type': 'tag'})

# Write JSON file
with open('source/_static/switcher.json', 'w') as f:
    json.dump(versions, f, indent=2)

print(f'Generated switcher.json with {len(versions)} versions')
if latest_stable:
    print(f'Latest stable release: {latest_stable}')

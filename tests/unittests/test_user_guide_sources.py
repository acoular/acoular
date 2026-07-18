from pathlib import Path
import re


DOCS_ROOT = Path(__file__).resolve().parents[2] / 'docs' / 'user_guide'
USER_GUIDE_PAGES = sorted(p for p in DOCS_ROOT.glob('*.rst') if p.name != 'index.rst')
LITERALINCLUDE_RE = re.compile(r'^\.\. literalinclude::\s+(?P<target>.+)$')


def test_user_guide_uses_example_literalincludes_for_runnable_python():
    """User guide pages should include runnable Python from examples, not inline blocks."""
    for page in USER_GUIDE_PAGES:
        text = page.read_text()
        assert '.. ipython::' not in text, f'{page} still contains ipython blocks'
        assert '.. code-block:: python' not in text, f'{page} still contains inline python blocks'
        assert '.. code:: python' not in text, f'{page} still contains inline python code directives'


def test_user_guide_literalincludes_point_to_examples():
    """All user guide code snippets should come from runnable example scripts."""
    for page in USER_GUIDE_PAGES:
        targets = []
        for line in page.read_text().splitlines():
            match = LITERALINCLUDE_RE.match(line.strip())
            if match:
                targets.append(match.group('target'))

        assert targets, f'{page} does not include any example snippets'

        for target in targets:
            assert target.startswith('../../examples/'), f'{page} includes non-example snippet: {target}'
            snippet_path = (page.parent / target).resolve()
            assert snippet_path.exists(), f'{page} includes missing file: {target}'
            assert snippet_path.name.startswith('example_'), f'{page} includes non-example file: {target}'

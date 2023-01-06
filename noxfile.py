
import nox

extras = ('', '[dev]', '[full]')

@nox.session
@nox.parametrize('extra', extras)
def build(session, extra):
    """Checks build for all supported Python versions and extras and optionally runs tests."""
    session.install(f'.{extra}')
    if extra == '[full]':
        with session.chdir('acoular/tests'):
            session.run('pip', 'list')
            session.run('/bin/bash', './run_tests.sh', external=True)

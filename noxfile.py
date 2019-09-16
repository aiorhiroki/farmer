import os
import nox


def default(session):
    """Default unit test session.
    """
    # Install all test dependencies, then install local packages in-place.
    session.install('mock', 'pytest', 'pytest-cov', 'pipenv')
    session.run(
        'pipenv',
        'install',
        '--dev')

    # Run py.test against the unit tests.
    session.run(
        'pipenv',
        'run',
        'python',
        '-m',
        'pytest',
        '--quiet',
        '--cov=tests',
        '--cov-append',
        '--cov-config=.coveragerc',
        '--cov-report=',
        '--cov-fail-under=0',
        os.path.join('tests'),
        *session.posargs
    )


@nox.session(python=['3.6'])
def unit(session):
    """Run the unit test suite."""
    default(session)


# @nox.session(python='3.6')
# def cover(session):
#     """Run the final coverage report.
#
#     This outputs the coverage report aggregating coverage from the unit
#     test runs (not system test runs), and then erases coverage data.
#     """
#     session.install('coverage', 'pytest-cov')
#     session.run('coverage', 'report', '--show-missing', '--fail-under=100')
#     session.run('coverage', 'erase')


@nox.session(python="3.6")
def black(session):
    """Run black.
    Format code to uniform standard.
    """
    session.install("black")
    session.run(
        "black",
        "apps",
        "tests",
        "--line-length=79"
    )


@nox.session(python='3.6')
def lint(session):
    """Run linters.

    Returns a failure if the linters find linting errors or sufficiently
    serious code quality issues.
    """

    session.install('flake8')
    session.run('flake8', os.path.join('apps'))
    session.run('flake8', 'tests')

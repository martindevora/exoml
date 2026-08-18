#!/bin/bash

set -e

CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_EXE}" ]]; then
  echo "Could not find conda. Set CONDA_EXE before running this script." >&2
  exit 1
fi
CONDA_BASE="$(${CONDA_EXE} info --base)"
export CONDA_EXE
export PATH="${CONDA_BASE}/condabin:${CONDA_BASE}/bin:${PATH}"

rm -f tests.log
rm -rf dist* .tox .pytest_cache build exoml-reqs *egg-info
conda env remove -n exoml-reqs -y 2>/dev/null || true
run_tests=false
git_tag=""
for arg in "$@"; do
  if [[ "${arg}" == "--tests" ]]; then
    run_tests=true
  elif [[ -z "${git_tag}" ]]; then
    git_tag="${arg}"
  else
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
done
if [[ -z "${git_tag}" ]]; then
  echo "Usage: $0 VERSION [--tests]" >&2
  exit 1
fi

echo "GIT TAG IS ${git_tag}"
if [[ "${run_tests}" == true ]]; then
  echo "Run all tests"
  python -m tox -r -e py311 > tests.log
  tests_results=$(grep "congratulations" tests.log || true)
  if [[ -z "${tests_results}" ]]; then
    echo "Tests failed; release aborted." >&2
    exit 1
  fi
else
  echo "Skipping tests (use --tests to run them)."
fi

rm -rf .tox .pytest_cache build exoml-reqs *egg-info
conda env remove -n exoml-reqs -y 2>/dev/null || true
conda create --override-channels -c conda-forge -n exoml-reqs python=3.11 -y
RELEASE_PYTHON="${CONDA_BASE}/envs/exoml-reqs/bin/python"
"${RELEASE_PYTHON}" -m pip install -U pip numpy==2.1.1
sed -i '16s/.*/version = "'${git_tag}'"/' pyproject.toml
"${RELEASE_PYTHON}" -m pip install -q build
"${RELEASE_PYTHON}" -m pip install .
git add pyproject.toml
git commit -m "Preparing release ${git_tag}"
git tag ${git_tag} -m "New release"
git push
git push --tags
set +e
rm -rf exoml-reqs dist* .tox .pytest_cache build *egg-info
conda env remove -n exoml-reqs -y 2>/dev/null || true
conda clean --all -y

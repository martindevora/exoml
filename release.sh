#!/bin/bash

set +e
rm tests.log
rm dist* -r
rm -r .tox
rm -r .pytest_cache
rm -r build
rm -r exoml-reqs
rm -R *egg-info
git_tag=$1
echo "GIT TAG IS " ${git_tag}
echo "Run all tests"
set -e
tox -r -e py310 > tests.log
tests_results=$(cat tests.log | grep "congratulations")
if ! [[ -z ${tests_results} ]]; then
  set +e
  rm tests.log
  rm -r .tox
  rm -r .pytest_cache
  rm -r build
  rm -r exoml-reqs
  rm -R *egg-info
  set -e
  python3 -m venv exoml-reqs
  source exoml-reqs/bin/activate
  python3 -m pip install pip -U
  python3 -m pip install numpy==1.23.5
  sed -i '16s/.*/version = "'${git_tag}'"/' pyproject.toml
  python3 -m pip install -q build
  python3 -m build
  deactivate
  git add pyproject.toml
  git commit -m "Preparing release ${git_tag}"
  git tag ${git_tag} -m "New release"
  git push
  git push --tags
else
  echo "TESTS FAILED. See tests.log"
fi
set +e
rm -R exoml-reqs
rm dist* -r
rm -r .tox
rm -r .pytest_cache
rm -r build
rm -R *egg-info
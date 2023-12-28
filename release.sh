#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
set +e
rm tests.log
rm dist* -r
rm -r .tox
rm -r .pytest_cache
rm -r build
rm -r exoml-reqs
rm -R *egg-info
conda remove -n exoml-reqs --all -y
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
  conda remove -n exoml-reqs --all -y
  set -e
  conda create -n exoml-reqs python=3.10 -y
  conda activate exoml-reqs
  python3 -m pip install pip -U
  python3 -m pip install numpy==1.23.5
  sed -i '16s/.*/version = "'${git_tag}'"/' pyproject.toml
  python3 -m pip install -q build
  python3 -m pip install .
  conda deactivate
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
conda remove -n exoml-reqs --all -y

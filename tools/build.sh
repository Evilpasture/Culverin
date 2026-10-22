#!/usr/bin/env bash

set -e 
set -o pipefail

python tools/build_project.py && pytest tests

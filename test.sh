#!/bin/bash
set -euo pipefail
docker buildx build --target test .

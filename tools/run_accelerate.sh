#!/usr/bin/env bash
set -e
accelerate launch --mixed_precision fp16 "$@"
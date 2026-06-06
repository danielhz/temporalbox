AISA        = aisa
AISA_DIR    = ~/nesy-ltel-code
LOCAL_DIR   = $(shell pwd)

RSYNC_OPTS  = --archive --compress --exclude='.git' \
              --exclude='__pycache__' --exclude='*.pyc' \
              --exclude='.pytest_cache' --exclude='.ruff_cache' \
              --exclude='.venv' --exclude='*.egg-info'

.PHONY: sync-push sync-pull test lint

## Push local code to aisa (excluding results/).
sync-push:
	rsync $(RSYNC_OPTS) --exclude='results/' $(LOCAL_DIR)/ $(AISA):$(AISA_DIR)/

## Pull results/ back from aisa.
sync-pull:
	rsync --archive --compress $(AISA):$(AISA_DIR)/results/ $(LOCAL_DIR)/results/

## Push, then immediately pull logs (useful after a quick run).
sync-roundtrip: sync-push
	rsync --archive --compress $(AISA):$(AISA_DIR)/logs/ $(LOCAL_DIR)/logs/

## Run local tests (no GPU required).
test:
	python -m pytest tests/ -v

## Lint with ruff.
lint:
	ruff check src/ tests/

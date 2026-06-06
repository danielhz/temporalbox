AISA        = aisa
AISA_DIR    = ~/nesy-ltel-code

# kg1 is the backup server — mirror results here because aisa goes into maintenance.
KG1         = kg1
KG1_DIR     = ~/nesy-ltel-backup/nesy-ltel

LOCAL_DIR   = $(shell pwd)

RSYNC_OPTS  = --archive --compress --exclude='.git' \
              --exclude='__pycache__' --exclude='*.pyc' \
              --exclude='.pytest_cache' --exclude='.ruff_cache' \
              --exclude='.venv' --exclude='*.egg-info'

.PHONY: sync-push sync-pull sync-backup sync-roundtrip test lint

## Push local code to aisa (excluding results/).
sync-push:
	rsync $(RSYNC_OPTS) --exclude='results/' $(LOCAL_DIR)/ $(AISA):$(AISA_DIR)/

## Pull results/ from aisa to local.
sync-pull:
	rsync --archive --compress $(AISA):$(AISA_DIR)/results/ $(LOCAL_DIR)/results/

## Mirror results/ from aisa → local → kg1 (run after sync-pull).
sync-backup: sync-pull
	ssh $(KG1) "mkdir -p $(KG1_DIR)/results"
	rsync --archive --compress $(LOCAL_DIR)/results/ $(KG1):$(KG1_DIR)/results/

## Push code, pull results, backup to kg1 in one go.
sync-roundtrip: sync-push sync-backup
	rsync --archive --compress $(AISA):$(AISA_DIR)/logs/ $(LOCAL_DIR)/logs/

## Run local tests (no GPU required).
test:
	python -m pytest tests/ -v

## Lint with ruff.
lint:
	ruff check src/ tests/

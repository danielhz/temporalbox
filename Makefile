# Host aliases for the GPU cluster (training) and the backup/data server.
# Set the real SSH host names in a gitignored `local.mk` (or the environment):
#   CLUSTER = <ssh-alias-of-gpu-cluster>
#   BACKUP  = <ssh-alias-of-backup-server>
CLUSTER    ?= cluster
BACKUP     ?= backup
-include local.mk

CLUSTER_DIR = ~/nesy-ltel-code
BACKUP_DIR  = ~/nesy-ltel-backup/nesy-ltel
LOCAL_DIR   = $(shell pwd)

RSYNC_OPTS  = --archive --compress --exclude='.git' \
              --exclude='__pycache__' --exclude='*.pyc' \
              --exclude='.pytest_cache' --exclude='.ruff_cache' \
              --exclude='.venv' --exclude='*.egg-info'

.PHONY: sync-push sync-pull sync-backup sync-roundtrip test lint

## Push local code to the GPU cluster (excluding results/).
sync-push:
	rsync $(RSYNC_OPTS) --exclude='results/' $(LOCAL_DIR)/ $(CLUSTER):$(CLUSTER_DIR)/

## Pull results/ from the GPU cluster to local.
sync-pull:
	rsync --archive --compress $(CLUSTER):$(CLUSTER_DIR)/results/ $(LOCAL_DIR)/results/

## Mirror results/ from the cluster -> local -> backup server (run after sync-pull).
sync-backup: sync-pull
	ssh $(BACKUP) "mkdir -p $(BACKUP_DIR)/results"
	rsync --archive --compress $(LOCAL_DIR)/results/ $(BACKUP):$(BACKUP_DIR)/results/

## Push code, pull results, mirror to backup in one go.
sync-roundtrip: sync-push sync-backup
	rsync --archive --compress $(CLUSTER):$(CLUSTER_DIR)/logs/ $(LOCAL_DIR)/logs/

## Run local tests (no GPU required).
test:
	python -m pytest tests/ -v

## Lint with ruff.
lint:
	ruff check src/ tests/

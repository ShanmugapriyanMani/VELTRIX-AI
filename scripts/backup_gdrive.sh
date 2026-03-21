#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# VELTRIX — Daily Google Drive Backup via rclone
# Runs daily at 15:45 IST (after ML retrain at 15:30)
# Usage: ./scripts/backup_gdrive.sh
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Config ──
REMOTE="gdrive"                              # rclone remote name
REMOTE_BASE="VELTRIX/backups"                # Google Drive folder path
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TODAY=$(date +%Y-%m-%d)
REMOTE_PATH="${REMOTE}:${REMOTE_BASE}/${TODAY}"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/backup.log"
KEEP_DAYS=30

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "═══ BACKUP START: ${TODAY} ═══"

# ── Pre-flight: check rclone + remote ──
if ! command -v rclone &>/dev/null; then
    log "ERROR: rclone not found in PATH"
    exit 1
fi

if ! rclone listremotes 2>/dev/null | grep -q "^${REMOTE}:$"; then
    log "ERROR: rclone remote '${REMOTE}' not configured. Run: rclone config"
    exit 1
fi

# ── Backup files ──
ERRORS=0

backup_file() {
    local src="$1"
    local dest_name="${2:-$(basename "$src")}"
    if [ -e "$src" ]; then
        if rclone copyto "$src" "${REMOTE_PATH}/${dest_name}" --log-level ERROR 2>>"$LOG_FILE"; then
            log "  OK: ${dest_name} ($(du -sh "$src" 2>/dev/null | cut -f1))"
        else
            log "  FAIL: ${dest_name}"
            ERRORS=$((ERRORS + 1))
        fi
    else
        log "  SKIP: ${src} (not found)"
    fi
}

backup_dir() {
    local src="$1"
    local dest_name="${2:-$(basename "$src")}"
    if [ -d "$src" ]; then
        if rclone copy "$src" "${REMOTE_PATH}/${dest_name}" --log-level ERROR 2>>"$LOG_FILE"; then
            local count
            count=$(find "$src" -type f | wc -l)
            log "  OK: ${dest_name}/ (${count} files)"
        else
            log "  FAIL: ${dest_name}/"
            ERRORS=$((ERRORS + 1))
        fi
    else
        log "  SKIP: ${src}/ (not found)"
    fi
}

# Primary: database
backup_file "${PROJECT_DIR}/data/trading_bot.db"

# ML models
backup_dir "${PROJECT_DIR}/models" "models"

# Instrument mappings
backup_file "${PROJECT_DIR}/data/fo_key_map.json"

# Circuit breaker state
backup_file "${PROJECT_DIR}/data/circuit_breaker_state.json"

# Config (no secrets — .env.plus has no API keys)
backup_file "${PROJECT_DIR}/.env.plus"

# Config YAML
backup_file "${PROJECT_DIR}/config/strategies.yaml"

# Logs (today's log only to save space)
TODAY_LOG="${PROJECT_DIR}/logs/bot_${TODAY}.log"
if [ -f "$TODAY_LOG" ]; then
    backup_file "$TODAY_LOG" "logs/bot_${TODAY}.log"
fi

# ── Prune old backups (keep last N days) ──
log "Pruning backups older than ${KEEP_DAYS} days..."
CUTOFF_DATE=$(date -d "${KEEP_DAYS} days ago" +%Y-%m-%d 2>/dev/null || date -v-${KEEP_DAYS}d +%Y-%m-%d 2>/dev/null)
if [ -n "$CUTOFF_DATE" ]; then
    rclone lsd "${REMOTE}:${REMOTE_BASE}" 2>/dev/null | awk '{print $NF}' | while read -r dir; do
        if [[ "$dir" < "$CUTOFF_DATE" ]]; then
            if rclone purge "${REMOTE}:${REMOTE_BASE}/${dir}" 2>>"$LOG_FILE"; then
                log "  PRUNED: ${dir}"
            fi
        fi
    done
fi

# ── Result ──
if [ "$ERRORS" -eq 0 ]; then
    log "═══ BACKUP COMPLETE: ${TODAY} (0 errors) ═══"
    exit 0
else
    log "═══ BACKUP FAILED: ${TODAY} (${ERRORS} errors) ═══"
    exit 1
fi

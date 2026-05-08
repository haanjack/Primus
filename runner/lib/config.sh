#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Configuration file loader for Primus CLI
# Supports: ~/.primusrc (shell format) and .primus.yaml (YAML format)
# Priority: CLI args > Project config > Global config > Defaults
#

# Requires common.sh to be sourced
if [[ -z "${__PRIMUS_COMMON_SOURCED:-}" ]]; then
    echo "[ERROR] config.sh requires common.sh to be sourced first" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Configuration Variables
# ---------------------------------------------------------------------------
# Always ensure associative array exists
if ! declare -p PRIMUS_CONFIG &>/dev/null; then
    declare -A PRIMUS_CONFIG
fi

PRIMUS_ROOT_DIR="${PRIMUS_ROOT_DIR:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../../" && pwd)}"
PRIMUS_RUNNER_DIR="${PRIMUS_RUNNER_DIR:-${PRIMUS_ROOT_DIR}/runner}"


# ---------------------------------------------------------------------------
# Guard: avoid duplicate sourcing
# ---------------------------------------------------------------------------
if [[ -n "${__PRIMUS_CONFIG_SOURCED:-}" ]]; then
  return 0
fi
export __PRIMUS_CONFIG_SOURCED=1

# Cache configuration
PRIMUS_CONFIG_CACHE_TTL=3600  # Cache validity in seconds (1 hour)

# ---------------------------------------------------------------------------
# Check if cached config is valid
# ---------------------------------------------------------------------------
is_cache_valid() {
    local config_file="$1"
    local cache_file="$2"

    LOG_DEBUG "Checking cache validity: $cache_file"

    # Cache doesn't exist
    if [[ ! -f "$cache_file" ]]; then
        LOG_DEBUG "Cache file does not exist"
        return 1
    fi

    # Source file is newer than cache
    if [[ "$config_file" -nt "$cache_file" ]]; then
        LOG_DEBUG "Source file is newer than cache"
        return 1
    fi

    # Cache is older than TTL
    local cache_age=$(($(date +%s) - $(stat -c %Y "$cache_file" 2>/dev/null || echo 0)))
    if [[ $cache_age -gt $PRIMUS_CONFIG_CACHE_TTL ]]; then
        LOG_DEBUG "Cache expired (age: ${cache_age}s, TTL: ${PRIMUS_CONFIG_CACHE_TTL}s)"
        return 1
    fi

    LOG_DEBUG "Cache is valid (age: ${cache_age}s)"
    return 0
}

# ---------------------------------------------------------------------------
# Load cached config
# ---------------------------------------------------------------------------
load_cache() {
    local cache_file="$1"

    LOG_DEBUG "Attempting to load cache from: $cache_file"

    if [[ -f "$cache_file" ]]; then
        # shellcheck disable=SC1090  # Dynamic source path
        if source "$cache_file" 2>/dev/null; then
            LOG_DEBUG "Successfully loaded cache from: $cache_file"
            return 0
        else
            LOG_DEBUG "Failed to load cache from: $cache_file"
        fi
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Save config to cache
# ---------------------------------------------------------------------------
save_cache() {
    local cache_file="$1"

    mkdir -p "$(dirname "$cache_file")" 2>/dev/null || return 1

    # Export current config to cache file
    {
        echo "# Primus CLI config cache"
        echo "# Generated: $(date)"
        echo ""
        for key in "${!PRIMUS_CONFIG[@]}"; do
            echo "PRIMUS_CONFIG[$key]=\"${PRIMUS_CONFIG[$key]}\""
        done
    } > "$cache_file" 2>/dev/null

    LOG_DEBUG "Config cached to: $cache_file"
}

# ---------------------------------------------------------------------------
# Load YAML Config File (.primus.yaml)
# ---------------------------------------------------------------------------
load_yaml_config() {
    local config_file="$1"

    LOG_DEBUG_RANK0 "Loading YAML config: $config_file"

    if [[ ! -f "$config_file" ]]; then
        LOG_ERROR_RANK0 "YAML config file not found: $config_file"
        return 1
    fi

    LOG_DEBUG_RANK0 "Parsing YAML file: $config_file"

    # Simple YAML parser (handles basic key: value format, arrays, and nested sections)
    local current_section=""
    local current_subsection=""
    local current_array_key=""
    local array_index=0
    local line_count=0

    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_count++))
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// /}" ]] && continue

        # Detect top-level section (e.g., "container:")
        if [[ "$line" =~ ^([a-z_]+):[[:space:]]*$ ]]; then
            current_section="${BASH_REMATCH[1]}"
            current_subsection=""
            current_array_key=""
            array_index=0
            LOG_DEBUG_RANK0 "Found section: $current_section (line $line_count)"
            continue
        fi

        # Detect subsection with 2 spaces (e.g., "  options:")
        if [[ "$line" =~ ^[[:space:]]{2}([a-z_-]+):[[:space:]]*$ ]]; then
            current_subsection="${BASH_REMATCH[1]}"
            current_array_key=""
            array_index=0
            LOG_DEBUG_RANK0 "Found subsection: $current_section.$current_subsection (line $line_count)"
            continue
        fi

        # Parse array item (e.g., "    - /data:/data" or "      - /dev/kfd")
        if [[ "$line" =~ ^[[:space:]]+\-[[:space:]]+(.+)$ ]]; then
            local value="${BASH_REMATCH[1]}"

            # Remove quotes
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"

            if [[ -n "$current_section" ]]; then
                if [[ -n "$current_subsection" ]] && [[ -n "$current_array_key" ]]; then
                    # Nested array under subsection (e.g., container.options.devices)
                    local config_key="${current_section}.${current_subsection}.${current_array_key}"
                    if [[ -z "${PRIMUS_CONFIG[$config_key]:-}" ]]; then
                        PRIMUS_CONFIG[$config_key]="$value"
                    else
                        PRIMUS_CONFIG[$config_key]+=$'\n'"$value"
                    fi
                    LOG_DEBUG_RANK0 "  $config_key += $value"
                    ((array_index++))
                elif [[ -n "$current_subsection" ]]; then
                    # Array directly under subsection
                    local config_key="${current_section}.${current_subsection}"
                    if [[ -z "${PRIMUS_CONFIG[$config_key]:-}" ]]; then
                        PRIMUS_CONFIG[$config_key]="$value"
                    else
                        PRIMUS_CONFIG[$config_key]+=$'\n'"$value"
                    fi
                    LOG_DEBUG_RANK0 "  $config_key += $value"
                    ((array_index++))
                elif [[ -n "$current_array_key" ]]; then
                    # Array directly under section
                    local config_key="${current_section}.${current_array_key}"
                    if [[ -z "${PRIMUS_CONFIG[$config_key]:-}" ]]; then
                        PRIMUS_CONFIG[$config_key]="$value"
                    else
                        PRIMUS_CONFIG[$config_key]+=$'\n'"$value"
                    fi
                    LOG_DEBUG_RANK0 "  $config_key += $value"
                    ((array_index++))
                fi
            fi
            continue
        fi

        # Parse nested key-value with 4 spaces (e.g., "    cpus: 16" or "    devices:")
        if [[ "$line" =~ ^[[:space:]]{4}([a-z_-]+):[[:space:]]*(.*)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"

            # Remove quotes
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"

            if [[ -n "$current_section" ]] && [[ -n "$current_subsection" ]]; then
                # If value is empty, this is an array key (e.g., "devices:")
                if [[ -z "$value" || "$value" == "[]" ]]; then
                    current_array_key="$key"
                    array_index=0
                    LOG_DEBUG_RANK0 "Found array key: $current_section.$current_subsection.$current_array_key (line $line_count)"
                    if [[ "$value" == "[]" ]]; then
                        local config_key="${current_section}.${current_subsection}.${key}"
                        PRIMUS_CONFIG[$config_key]="$value"
                        LOG_DEBUG_RANK0 "  $config_key = $value"
                    fi
                else
                    # Regular key-value pair
                    current_array_key=""
                    local config_key="${current_section}.${current_subsection}.${key}"
                    PRIMUS_CONFIG[$config_key]="$value"
                    LOG_DEBUG_RANK0 "  $config_key = $value"
                fi
            fi
            continue
        fi

        # Parse key-value with 2 spaces (e.g., "  image: value")
        if [[ "$line" =~ ^[[:space:]]{2}([a-z_-]+):[[:space:]]*(.+)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"

            # Remove quotes
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"

            # Reset subsection when we get a normal key-value pair
            current_subsection=""
            current_array_key=""
            array_index=0

            if [[ -n "$current_section" ]]; then
                local config_key="${current_section}.${key}"
                PRIMUS_CONFIG[$config_key]="$value"
                LOG_DEBUG_RANK0 "  $config_key = $value"
            fi
        fi
    done < "$config_file"

    LOG_DEBUG_RANK0 "Loaded YAML config: $config_file (processed $line_count lines)"
    LOG_DEBUG_RANK0 "Total config keys loaded: ${#PRIMUS_CONFIG[@]}"

    return 0
}

# ---------------------------------------------------------------------------
# Resolve Config File Path
# Priority:
#   1) Explicit CLI config file (argument to load_config_auto)
#   2) User global config (~/.primus.yaml)
#   3) System default config (runner/.primus.yaml)
#
# Returns the first candidate path via stdout; non-zero exit code if none found.
# Note: Existence of the CLI path is validated later by load_yaml_config.
# ---------------------------------------------------------------------------
resolve_config_file() {
    local cli_config_file="${1:-}"
    local global_config="$HOME/.primus.yaml"
    local system_config="${PRIMUS_RUNNER_DIR}/.primus.yaml"

    # 1) Explicit CLI override (may or may not exist; load_yaml_config will validate)
    if [[ -n "$cli_config_file" ]]; then
        echo "$cli_config_file"
        return 0
    fi

    # 2) User-global config
    if [[ -f "$global_config" ]]; then
        echo "$global_config"
        return 0
    fi

    # 3) System default config
    if [[ -f "$system_config" ]]; then
        echo "$system_config"
        return 0
    fi

    # No config file found
    return 1
}

# ---------------------------------------------------------------------------
# Load Config Automatically (with CLI override support)
# Usage: load_config_auto [config_file] [log_prefix]
#
# If a config file is resolved (CLI/user/system), it will be loaded; otherwise
# the function returns successfully without loading any config.
# ---------------------------------------------------------------------------
load_config_auto() {
    local config_file="${1:-}"
    local log_prefix="${2:-main}"

    local resolved_cfg
    if ! resolved_cfg="$(resolve_config_file "$config_file")"; then
        LOG_INFO_RANK0 "[$log_prefix] No configuration file found (CLI/user/system), continuing with defaults."
        return 0
    fi

    load_yaml_config "$resolved_cfg" || {
        LOG_ERROR "[$log_prefix] Failed to load config: $resolved_cfg"
        return 1
    }

    LOG_INFO_RANK0 "[$log_prefix] Loaded config: $resolved_cfg"

    return 0
}

# ---------------------------------------------------------------------------
# Get Config Value
# ---------------------------------------------------------------------------
get_config() {
    local key="$1"
    local default="${2:-}"

    if [[ -n "${PRIMUS_CONFIG[$key]:-}" ]]; then
        LOG_DEBUG "get_config: $key = ${PRIMUS_CONFIG[$key]}"
        echo "${PRIMUS_CONFIG[$key]}"
    else
        LOG_DEBUG "get_config: $key not found, using default = $default"
        echo "$default"
    fi
}

# ---------------------------------------------------------------------------
# Set Config Value (override)
# ---------------------------------------------------------------------------
set_config() {
    local key="$1"
    local value="$2"

    PRIMUS_CONFIG[$key]="$value"
    LOG_DEBUG "Config set: $key = $value"
}

# ---------------------------------------------------------------------------
# Extract Config Section
# Extract all config keys matching a prefix and remove the prefix
# Usage: extract_config_section "slurm" result_array
# ---------------------------------------------------------------------------
extract_config_section() {
    local prefix="$1"
    # shellcheck disable=SC2034  # result_array is used via nameref
    local -n result_array="$2"  # nameref to associative array

    LOG_DEBUG "Extracting config section: $prefix"

    local count=0
    # Extract all config keys matching prefix and remove the prefix
    for key in "${!PRIMUS_CONFIG[@]}"; do
        if [[ "$key" =~ ^${prefix}\. ]]; then
            # Remove prefix to get parameter name (e.g., "slurm.partition" -> "partition")
            local param_name="${key#"${prefix}".}"
            # shellcheck disable=SC2034  # result_array is a nameref, accessed indirectly
            result_array["$param_name"]="${PRIMUS_CONFIG[$key]}"
            ((count++))
            LOG_DEBUG "  Extracted: $param_name = ${PRIMUS_CONFIG[$key]}"
        fi
    done

    LOG_DEBUG "Extracted $count config entries for section: $prefix"

    return 0
}
export -f extract_config_section

# ---------------------------------------------------------------------------
# Pretty-print Config Section (for debugging)
# Usage: print_config_section "section_name" associative_array
# ---------------------------------------------------------------------------
print_config_section() {
    local section="$1"
    # shellcheck disable=SC2034
    local -n section_array="$2"  # name reference to associative array

    print_section "[$section] Config Section"  # uses print_section from common.sh

    for key in "${!section_array[@]}"; do
        PRINT_INFO_RANK0 "  $key = ${section_array[$key]}"
    done
}
export -f print_config_section

# ---------------------------------------------------------------------------
# Resolve target_gpu from Primus args array.
# Checks (in order):
#   1. Bare arg `target_gpu=VALUE` in the args array
#   2. `target_gpu:` key in the training YAML passed via --config
# Prints "cuda", "amd", or "auto" (default).
# Usage: resolve_target_gpu ARRAY_NAME   (pass array name, not value)
# ---------------------------------------------------------------------------
resolve_target_gpu() {
    local -n _rta_args="$1"
    local _rta_result="auto"

    # 1) bare arg: target_gpu=VALUE
    local _arg
    for _arg in "${_rta_args[@]}"; do
        if [[ "$_arg" =~ ^target_gpu=(.+)$ ]]; then
            _rta_result="${BASH_REMATCH[1]}"
            echo "$_rta_result"
            return 0
        fi
    done

    # 2) --config <yaml> path: grep target_gpu
    local _rta_cfg=""
    local _rta_i
    for (( _rta_i=0; _rta_i<${#_rta_args[@]}; _rta_i++ )); do
        if [[ "${_rta_args[$_rta_i]}" == "--config" ]]; then
            _rta_cfg="${_rta_args[$((_rta_i+1))]:-}"
            break
        fi
    done
    if [[ -n "$_rta_cfg" ]] && [[ -f "$_rta_cfg" ]]; then
        local _rta_val
        _rta_val=$(grep -E '^target_gpu:[[:space:]]*' "$_rta_cfg" 2>/dev/null \
                   | head -1 | awk -F':[[:space:]]*' '{print $2}' \
                   | tr -d '"'"'" | xargs 2>/dev/null || true)
        if [[ -n "$_rta_val" ]]; then
            _rta_result="$_rta_val"
        fi
    fi

    echo "$_rta_result"
}
export -f resolve_target_gpu

# ---------------------------------------------------------------------------
# Load GPU-specific overlay config (cuda.yaml) on top of the current config.
# Called after load_config_auto; merges cuda.yaml when NVIDIA is detected.
# Condition: target_gpu="cuda"  OR  (target_gpu="auto" AND nvidia-smi present
#            AND rocm-smi absent)
# Usage: load_gpu_overlay_config "cuda"|"amd"|"auto"
# ---------------------------------------------------------------------------
load_gpu_overlay_config() {
    local _lgo_target="${1:-auto}"
    # Prefer RUNNER_DIR already resolved by the calling script; fall back to
    # BASH_SOURCE-relative path as a last resort (may be unreliable when
    # config.sh is sourced via a relative path inside a container).
    local _lgo_runner_dir
    if [[ -n "${RUNNER_DIR:-}" ]]; then
        _lgo_runner_dir="$RUNNER_DIR"
    else
        _lgo_runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi

    local _lgo_is_cuda=false
    if [[ "$_lgo_target" == "cuda" ]]; then
        _lgo_is_cuda=true
    elif [[ "$_lgo_target" == "auto" ]]; then
        if command -v nvidia-smi &>/dev/null && ! command -v rocm-smi &>/dev/null; then
            _lgo_is_cuda=true
        fi
    fi

    if [[ "$_lgo_is_cuda" == "true" ]]; then
        local _lgo_overlay="${_lgo_runner_dir}/cuda.yaml"
        if [[ -f "$_lgo_overlay" ]]; then
            load_yaml_config "$_lgo_overlay" || {
                LOG_WARN "[config] Failed to merge GPU overlay: $_lgo_overlay"
                return 0
            }
            LOG_INFO_RANK0 "[config] Auto-merged GPU overlay: $_lgo_overlay"
        else
            LOG_WARN "[config] GPU overlay not found: $_lgo_overlay"
        fi
    fi
}
export -f load_gpu_overlay_config

# ---------------------------------------------------------------------------
# Export all functions
# ---------------------------------------------------------------------------
export -f load_yaml_config load_config_auto resolve_config_file
export -f get_config set_config

LOG_DEBUG_RANK0 "Primus config library loaded successfully"

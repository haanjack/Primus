#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# When MASTER_ADDR is set to a real network IP (not localhost), prefer the
# local interface that owns that IP so that gloo/NCCL bind to the correct
# interface on multi-homed hosts.
if [[ -n "${MASTER_ADDR:-}" ]] && [[ "${MASTER_ADDR}" != "localhost" ]]; then
    _master_ip=$(getent hosts "${MASTER_ADDR}" 2>/dev/null | awk '{print $1}' | head -1)
    if [[ -n "$_master_ip" ]]; then
        _iface=$(ip -o -4 addr show | awk -v ip="$_master_ip" '$4 ~ ip {print $2}' | head -1)
        if [[ -n "$_iface" ]]; then
            echo "$_iface"
            exit 0
        fi
    fi
fi

# Fallback: use the interface bound to the first IP reported by hostname -I
IP_INTERFACE=$(ip -o -4 addr show | awk -v ip="$(hostname -I | awk '{print $1}')" '$4 ~ ip {print $2}')

if [[ -z "$IP_INTERFACE" ]]; then
    echo "Error: No active ip interface found!" >&2
    exit 1
fi

# Print the result (to be captured by calling script)
echo "$IP_INTERFACE"

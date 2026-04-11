#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh — wrapper to fix libgobject symbol conflict in Conda + Isaac Sim
#
# Root cause:
#   Isaac Sim bundles its own libgobject-2.0.so.0 inside the extscache dir.
#   When launched from a Conda env, that bundled lib is missing the symbol
#   `g_string_copy` (compiled against a different glib version), causing:
#     "GpuFoundationFactory: IGpuFoundationFactory interface not found"
#
# Fix:
#   Preload the system-provided glib/gobject before the process starts so
#   the dynamic linker resolves all symbols from the correct version.
# ---------------------------------------------------------------------------

export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libglib-2.0.so.0:/usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0"

exec python sim_main.py "$@"

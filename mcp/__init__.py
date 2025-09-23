"""MCP servers and helpers for gymnasium-solver.

Currently exposes :mod:`mcp.debug_server` which offers tooling to inspect
training runs and environments via the Model Context Protocol (JSON-RPC over
stdio).
"""

__all__ = ["debug_server"]

"""
IRH Desktop - Plugin System

Provides a plugin architecture for extending IRH Desktop functionality:
- Plugin base class and API
- Plugin discovery and loading
- Plugin management
- Security sandboxing

This implements Phase 6 of the DEB_PACKAGE_ROADMAP.md.

Author: Brandon D. McCrary
"""

from irh_desktop.plugins.base import (
    IRHPlugin,
    PluginInfo,
    PluginContext,
    PluginResult,
)
from irh_desktop.plugins.manager import (
    PluginManager,
    discover_plugins,
    load_plugin,
)

__all__ = [
    # Base classes
    "IRHPlugin",
    "PluginInfo",
    "PluginContext",
    "PluginResult",
    # Manager
    "PluginManager",
    "discover_plugins",
    "load_plugin",
]

"""
User interface components for AutoAUC.
"""

from .gui import AutoAUCApp, Parameter, Space
from .cli import main as cli_main
from .gui_main import main as gui_main
from .monitor import MonitorPage
from .session_manager import get_session_manager, SessionManager, SessionInfo

__all__ = [
    "AutoAUCApp",
    "Parameter", 
    "Space",
    "cli_main",
    "gui_main",
    "create_monitor_page",
    "MonitorPage",
    "get_session_manager",
    "SessionManager",
    "SessionInfo",
]

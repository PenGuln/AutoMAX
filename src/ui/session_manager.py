"""
Session Manager for AutoAUC Training Sessions.
Handles persistent session tracking, status monitoring, and log management.
"""

import json
import os
import time
import threading
import psutil
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class SessionInfo:
    """Information about a training session."""
    session_id: str
    name: str
    status: str  # 'running', 'completed', 'failed', 'cancelled'
    start_time: str
    end_time: Optional[str] = None
    model: str = ""
    dataset: str = ""
    target: str = ""
    epochs: int = 0
    current_epoch: int = 0
    progress: float = 0.0
    loss: float = 0.0
    config: Dict[str, Any] = None
    log_file: str = ""
    output_path: str = ""
    pid: Optional[int] = None
    last_heartbeat: Optional[str] = None
    error_message: Optional[str] = None
    test_true: Optional[List[float]] = None  # True labels from last epoch
    test_pred: Optional[List[float]] = None  # Predictions from last epoch
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class SessionManager:
    """Manages training sessions with persistence and real-time monitoring."""
    
    def __init__(self, sessions_file: str = "sessions.json", logs_dir: str = "logs"):
        logger.info(f"Initializing SessionManager with sessions_file: {sessions_file} and logs_dir: {logs_dir}")
        self.sessions_file = sessions_file
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.sessions: Dict[str, SessionInfo] = {}
        self._lock = threading.RLock()  # Thread safety for in-memory operations
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 5  # seconds
        
        # GUI callback for progress updates
        self.gui_callback = None
        self.gui_current_session_id = None
        self._last_gui_update = None  # Cache last GUI update info
        
        # Load existing sessions
        self._load_sessions()
        
        # Start monitoring thread
        self.start_monitoring()
    
    def _load_sessions(self):
        """Load sessions from disk."""
        if not os.path.exists(self.sessions_file):
            return
            
        try:
            with open(self.sessions_file, 'r') as f:
                data = json.load(f)
            
            with self._lock:
                self.sessions.clear()
                for session_data in data["sessions"]:
                    session = SessionInfo(**session_data)
                    self.sessions[session.session_id] = session
                    
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
    
    def _save_sessions(self):
        """Save sessions to disk."""
        try:
            with self._lock:
                sessions_data = {
                    "sessions": []
                }
                for session in self.sessions.values():
                    session_dict = asdict(session)
                    # Convert numpy arrays to lists for JSON serialization
                    if session_dict.get('test_true') is not None:
                        if isinstance(session_dict['test_true'], np.ndarray):
                            session_dict['test_true'] = session_dict['test_true'].tolist()
                    if session_dict.get('test_pred') is not None:
                        if isinstance(session_dict['test_pred'], np.ndarray):
                            session_dict['test_pred'] = session_dict['test_pred'].tolist()
                    sessions_data["sessions"].append(session_dict)
            
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions_data, f, indent=2)
                
            logger.debug(f"Saved {len(self.sessions)} sessions to {self.sessions_file}")
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def create_session(self, name: str, config: Dict[str, Any], **kwargs) -> SessionInfo:
        """Create a new training session."""
        session_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()
        
        session = SessionInfo(
            session_id=session_id,
            name=name,
            status='running',
            start_time=current_time,
            model=config.get('model', ''),
            dataset=config.get('dataset', ''),
            target=config.get('target', ''),
            epochs=config.get('epochs', 0),
            config=config,
            log_file=str(self.logs_dir / f"session_{session_id}.log"),
            output_path=config.get('output_path', './output'),
            pid=os.getpid(),
            last_heartbeat=current_time
        )
        
        # Update with any additional kwargs
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)

        self._load_sessions()
        self.sessions[session_id] = session
        self._save_sessions()
        
        logger.info(f"Created session {session_id}: {name}")
        return session
    
    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session information with minimal I/O."""
        self._load_sessions()
        logger.info(f"Updating session {session_id} with kwargs: {kwargs.keys()}")
        
        with self._lock:
            if session_id not in self.sessions:
                return False
        
            session = self.sessions[session_id]
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            session.last_heartbeat = datetime.now().isoformat()
        
        # Save immediately for critical updates
        self._save_sessions()
        
        return True
    
    def complete_session(self, session_id: str, error_message: str = None):
        """Mark a session as completed or failed."""
        self._load_sessions()
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.status = 'failed' if error_message else 'completed'
        session.end_time = datetime.now().isoformat()
        
        if error_message:
            session.error_message = error_message
        
        self._save_sessions()
        
        status = "failed" if error_message else "completed"
        logger.info(f"Session {session_id} {status}")
        return True
    
    def cancel_session(self, session_id: str):
        """Cancel a running session."""
        self._load_sessions()
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if session.status == 'running':
            session.status = 'cancelled'
            session.end_time = datetime.now().isoformat()
            self._save_sessions()
            
            # Try to kill the process if PID is available
            if session.pid:
                try:
                    process = psutil.Process(session.pid)
                    process.terminate()
                    logger.info(f"Terminated process {session.pid} for session {session_id}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.warning(f"Could not terminate process {session.pid} for session {session_id}")
        
        logger.info(f"Cancelled session {session_id}")
        return True
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID with fresh data from disk."""
        self._load_sessions()
        with self._lock:
            return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> List[SessionInfo]:
        """Get all sessions with fresh data from disk."""
        self._load_sessions()
        with self._lock:
            return list(self.sessions.values())
    
    def get_running_sessions(self) -> List[SessionInfo]:
        """Get all running sessions with fresh data from disk."""
        self._load_sessions()
        with self._lock:
            return [s for s in self.sessions.values() if s.status == 'running']
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        self._load_sessions()
        if session_id not in self.sessions:
            return False
        
        # Cancel if running
        if self.sessions[session_id].status == 'running':
            self.cancel_session(session_id)
        
        del self.sessions[session_id]
        self._save_sessions()
        
        logger.info(f"Deleted session {session_id}")
        return True
    
    def _check_session_liveness(self, session: SessionInfo) -> bool:
        """Check if a session is still alive."""
        if session.status != 'running':
            return True
        
        # Check if process is still running
        if session.pid:
            try:
                process = psutil.Process(session.pid)
                if not process.is_running():
                    logger.warning(f"Process {session.pid} for session {session.session_id} is not running")
                    return False
            except psutil.NoSuchProcess:
                logger.warning(f"Process {session.pid} for session {session.session_id} not found")
                return False
        
        # Check heartbeat timeout (5 minutes)
        if session.last_heartbeat:
            try:
                last_heartbeat = datetime.fromisoformat(session.last_heartbeat)
                timeout = datetime.now().timestamp() - last_heartbeat.timestamp()
                if timeout > 300:  # 5 minutes
                    logger.warning(f"Session {session.session_id} heartbeat timeout")
                    return False
            except ValueError:
                pass
        
        return True
    
    def _monitor_sessions(self):
        """Monitor running sessions for liveness."""
        while self.monitoring:
            try:
                for session in self.get_running_sessions():
                    if not self._check_session_liveness(session):
                        # Mark as failed due to timeout/crash
                        self.complete_session(
                            session.session_id, 
                            error_message="Session timeout or process crashed"
                        )
                    else:
                        # Update GUI progress for current session
                        self._update_gui_progress(session)
                
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in session monitoring: {e}")
                time.sleep(self.monitor_interval)
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_sessions, daemon=True)
            self.monitor_thread.start()
            logger.info("Started session monitoring")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("Stopped session monitoring")
    
    def get_session_logs(self, session_id: str, lines: int = 100) -> List[str]:
        """Get recent logs for a session."""
        session = self.get_session(session_id)
        if not session or not session.log_file:
            return []
        
        try:
            if os.path.exists(session.log_file):
                with open(session.log_file, 'r', encoding='utf-8') as f:
                    all_lines = f.readlines()
                    return all_lines[-lines:] if lines > 0 else all_lines
        except Exception as e:
            logger.error(f"Error reading logs for session {session_id}: {e}")
        
        return []
    
    def write_session_log(self, session_id: str, message: str, level: str = "INFO"):
        """Write a log message to a session's log file."""
        session = self.get_session(session_id)
        if not session:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] {message}\n"
            
            with open(session.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            logger.error(f"Error writing log for session {session_id}: {e}")
    
    def heartbeat(self, session_id: str):
        """Send a heartbeat for a session."""
        if self.update_session(session_id, last_heartbeat=datetime.now().isoformat()):
            logger.debug(f"Heartbeat for session {session_id}")
    
    def register_gui_callback(self, gui_instance, current_session_id: str = None):
        """Register GUI instance for progress updates."""
        self.gui_callback = gui_instance
        self.gui_current_session_id = current_session_id
        self._last_gui_update = None  # Clear cache for new session
        logger.info(f"Registered GUI callback for session: {current_session_id}")
    
    def unregister_gui_callback(self):
        """Unregister GUI callback."""
        self.gui_callback = None
        self.gui_current_session_id = None
        self._last_gui_update = None  # Clear cache
        logger.info("Unregistered GUI callback")
    
    def _update_gui_progress(self, session: SessionInfo):
        """Update GUI progress if callback is registered and session matches."""
        if (self.gui_callback and 
            self.gui_current_session_id and 
            session.session_id == self.gui_current_session_id and
            hasattr(self.gui_callback, 'set_progressbar')):
            
            # Skip initial state (epoch 0, progress 0.0)
            if session.current_epoch == 0 and session.progress == 0.0:
                return
            
            # Create current session info tuple for comparison
            current_info = (session.current_epoch, session.epochs, session.progress, session.loss)
            
            # Skip update if session info is the same as last report
            if self._last_gui_update == current_info:
                return
            
            try:
                # Update progress bar
                self.gui_callback.set_progressbar(session.progress)
                
                # Update console with progress info
                if hasattr(self.gui_callback, 'console_log'):
                    self.gui_callback.console_log(
                        f"Epoch {session.current_epoch}/{session.epochs} - Progress: {session.progress:.1%} - Loss: {session.loss:.4f}",
                        "TRAINING"
                    )
                
                # Cache the current session info
                self._last_gui_update = current_info
                
            except Exception as e:
                logger.error(f"Error updating GUI progress: {e}")
    


# Global session manager instance
_session_manager = None

def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager

def cleanup_session_manager():
    """Cleanup the session manager."""
    global _session_manager
    if _session_manager:
        _session_manager.stop_monitoring()
        _session_manager = None


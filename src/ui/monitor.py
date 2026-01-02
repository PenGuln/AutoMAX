"""
Training session monitoring page for AutoAUC.
"""

import flet as ft
import os
import json
import glob
import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from sklearn.metrics import roc_curve
from .session_manager import get_session_manager, SessionInfo
import logging
logger = logging.getLogger(__name__)

# TrainingSession class removed - using SessionInfo from session_manager instead


class MonitorPage(ft.Column):
    """Training session monitoring page."""
    
    def __init__(self):
        super().__init__()
        # Use global session manager
        self.session_manager = get_session_manager()
        self.sessions: List[SessionInfo] = []
        self.selected_session = None
        
        # Initialize UI components
        self._create_header()
        self._create_session_list()
        self._create_session_details()
        self._create_controls()
        
        # Load existing sessions (defer updates until added to page)
        self._load_sessions_data()
        
        # Set up the main layout
        self.controls = [
            self.header,
            ft.Container(
                content=ft.Row([
                    self.session_list_container,
                    self.session_details_container
                ]),
                padding=ft.padding.all(20)
            ),
            self.controls_container
        ]
    
    def _create_header(self):
        """Create the page header."""
        self.header = ft.Container(
            content=ft.Row([
                ft.IconButton(
                    icon=ft.icons.ARROW_BACK,
                    tooltip="Back to Main",
                    on_click=self._go_back,
                    bgcolor=ft.colors.BLUE_100,
                    icon_color=ft.colors.BLUE_700,
                ),
                ft.Text(
                    "Training Session Monitor",
                    size=24,
                    weight=ft.FontWeight.BOLD,
                    color=ft.colors.GREY_800
                ),
                ft.IconButton(
                    icon=ft.icons.REFRESH,
                    tooltip="Refresh Sessions",
                    on_click=self._refresh_sessions,
                    bgcolor=ft.colors.GREEN_100,
                    icon_color=ft.colors.GREEN_700,
                ),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            padding=ft.padding.all(15),
            margin=ft.margin.only(bottom=15),
            bgcolor=ft.colors.GREY_50,
            border_radius=ft.border_radius.all(8),
        )
    
    def _create_session_list(self):
        """Create the session list view."""
        self.session_list = ft.ListView(
            height=400,
            width=400,
            spacing=5,
            padding=ft.padding.all(10),
        )
        
        self.session_list_container = ft.Container(
            content=ft.Column([
                ft.Text("Training Sessions", size=18, weight=ft.FontWeight.BOLD),
                self.session_list
            ]),
            width=400,
            height=500,
            border=ft.border.all(1, ft.colors.GREY_300),
            border_radius=ft.border_radius.all(8),
            padding=ft.padding.all(10),
        )
    
    def _create_session_details(self):
        """Create the session details view."""
        self.session_details = ft.Column([
            ft.Text("Session Details", size=18, weight=ft.FontWeight.BOLD),
            ft.Text("Select a session to view details", color=ft.colors.GREY_600)
        ], spacing=10, scroll=ft.ScrollMode.AUTO, expand=True)
        
        self.session_details_container = ft.Container(
            content=self.session_details,
            width=600,
            height=500,
            border=ft.border.all(1, ft.colors.GREY_300),
            border_radius=ft.border_radius.all(8),
            padding=ft.padding.all(15),
        )
    
    def _create_controls(self):
        """Create control buttons."""
        self.controls_container = ft.Container(
            content=ft.Row([
                ft.ElevatedButton(
                    "Cancel Session",
                    icon=ft.icons.STOP,
                    on_click=self._cancel_session,
                    disabled=True,
                    bgcolor=ft.colors.ORANGE_600,
                    color=ft.colors.WHITE,
                ),
                ft.ElevatedButton(
                    "View Logs",
                    icon=ft.icons.DESCRIPTION,
                    on_click=self._view_logs,
                    disabled=True,
                    bgcolor=ft.colors.GREEN_600,
                    color=ft.colors.WHITE,
                ),
                ft.ElevatedButton(
                    "Delete Session",
                    icon=ft.icons.DELETE,
                    on_click=self._delete_session,
                    disabled=True,
                    bgcolor=ft.colors.RED_600,
                    color=ft.colors.WHITE,
                ),
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=15),
            padding=ft.padding.all(15),
        )
    
    def _load_sessions_data(self):
        """Load training sessions data without updating UI."""
        
        self.sessions = self.session_manager.get_all_sessions()
    
    def _load_sessions(self):
        """Load training sessions and update UI."""
        self._load_sessions_data()
        self._update_session_list()
        self._update_session_details()
        self._update_controls()
    
    def _update_session_list(self):
        """Update the session list display."""
        self.session_list.controls = []
        
        if not self.sessions:
            self.session_list.controls.append(
                ft.Container(
                    content=ft.Text("No training sessions found", color=ft.colors.GREY_600),
                    alignment=ft.alignment.center,
                    height=100,
                )
            )
        else:
            for session in self.sessions:
                session_card = self._create_session_card(session)
                self.session_list.controls.append(session_card)
        
        self.session_list.update()
    
    def _create_session_card(self, session: SessionInfo):
        """Create a session card for the list."""
        status_color = {
            'running': ft.colors.GREEN_600,
            'completed': ft.colors.BLUE_600,
            'failed': ft.colors.RED_600,
            'cancelled': ft.colors.ORANGE_600,
            'Unknown': ft.colors.GREY_600
        }.get(session.status, ft.colors.GREY_600)
        
        # Format status for display
        status_display = session.status.title()
        if session.status == 'running':
            status_display = "🔄 Running"
        elif session.status == 'completed':
            status_display = "✅ Completed"
        elif session.status == 'failed':
            status_display = "❌ Failed"
        elif session.status == 'cancelled':
            status_display = "⏹️ Cancelled"
        
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text(f"Session: {session.session_id[:8]}...", weight=ft.FontWeight.BOLD),
                    ft.Container(
                        content=ft.Text(status_display, color=ft.colors.WHITE, size=12),
                        bgcolor=status_color,
                        padding=ft.padding.symmetric(4, 8),
                        border_radius=ft.border_radius.all(4),
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Text(f"Name: {session.name}", size=12, color=ft.colors.GREY_700),
                ft.Text(f"Model: {session.model}", size=12, color=ft.colors.GREY_700),
                ft.Text(f"Dataset: {session.dataset}", size=12, color=ft.colors.GREY_700),
                ft.Text(f"Target: {session.target}", size=12, color=ft.colors.GREY_700),
                ft.Text(f"Progress: {session.progress:.2f}", size=12, color=ft.colors.GREY_700),
            ], spacing=5),
            padding=ft.padding.all(10),
            border=ft.border.all(1, ft.colors.GREY_300),
            border_radius=ft.border_radius.all(6),
            bgcolor=ft.colors.WHITE,
            width=380,
            height=180,
            on_click=lambda e, s=session: self._select_session(s),
        )
    
    def _select_session(self, session: SessionInfo):
        """Select a session and show its details."""
        self.selected_session = session
        self._update_session_details()
        self._update_controls()
    
    def _update_session_details(self):
        """Update the session details display."""
        if not self.selected_session:
            self.session_details.controls = [
                ft.Text("Session Details", size=18, weight=ft.FontWeight.BOLD),
                ft.Text("Select a session to view details", color=ft.colors.GREY_600)
            ]
        else:
            session = self.selected_session
            progress_value = session.progress if session.progress > 0 else (session.current_epoch / session.epochs if session.epochs > 0 else 0)
            
            details = [
                ft.Text("Session Details", size=18, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Text(f"Session ID: {session.session_id}", weight=ft.FontWeight.BOLD),
                ft.Text(f"Name: {session.name}"),
                ft.Text(f"Status: {session.status.title()}"),
                ft.Text(f"Start Time: {session.start_time}"),
                ft.Text(f"End Time: {session.end_time or 'N/A'}"),
                ft.Text(f"Model: {session.model}"),
                ft.Text(f"Dataset: {session.dataset}"),
                ft.Text(f"Target: {session.target}"),
                ft.Text(f"Epochs: {session.epochs}"),
                ft.Text(f"Current Epoch: {session.current_epoch}"),
                ft.Text(f"Progress: {session.progress:.2f}"),
                ft.Text(f"Loss: {session.loss:.4f}"),
                # ft.ProgressBar(value=progress_value, width=300) if progress_value > 0 else ft.Text("Progress: N/A"),
            ]
            
            if session.error_message:
                details.append(ft.Text(f"Error: {session.error_message}", color=ft.colors.RED_600))
            
            if session.last_heartbeat:
                details.append(ft.Text(f"Last Heartbeat: {session.last_heartbeat}"))
            
            # Add ROC curve visualization if test_true and test_pred are available
            if session.test_true is not None and session.test_pred is not None:
                details.append(ft.Divider())
                details.append(ft.Text("ROC Curve", size=16, weight=ft.FontWeight.BOLD))
                roc_chart = self._create_roc_chart(session.test_true, session.test_pred)
                if roc_chart:
                    details.append(roc_chart)
            
            self.session_details.controls = details
        
        self.session_details.update()
    
    def _create_roc_chart(self, test_true: List[float], test_pred: List[float]) -> Optional[ft.LineChart]:
        """Create an ROC curve chart from test_true and test_pred."""
        try:
            # Convert to numpy arrays if needed
            if isinstance(test_true, list):
                test_true = np.array(test_true)
            if isinstance(test_pred, list):
                test_pred = np.array(test_pred)
            
            # Ensure predictions are in [0,1] range (should already be probabilities from evaluate)
            # But apply sigmoid if needed as a safety check
            if test_pred.min() < 0 or test_pred.max() > 1:
                from scipy.special import expit
                test_pred = expit(test_pred)
            
            # Flatten arrays if needed
            if test_pred.ndim > 1:
                test_pred = test_pred.flatten()
            if test_true.ndim > 1:
                test_true = test_true.flatten()
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(test_true, test_pred)
            
            # Create data points for the chart
            data_points = [
                ft.LineChartDataPoint(x, y) 
                for x, y in zip(fpr.tolist(), tpr.tolist())
            ]
            
            # Create chart data series
            roc_data = ft.LineChartData(
                data_points=data_points,
                stroke_width=2,
                color=ft.colors.BLUE,
                curved=True,
                stroke_cap_round=True,
            )
            
            # Create the chart with adjusted sizing to accommodate labels
            # Wrap chart in container to ensure proper spacing for labels
            chart = ft.Container(
                content=ft.LineChart(
                    data_series=[roc_data],
                    border=ft.border.all(1, ft.colors.GREY_400),
                    left_axis=ft.ChartAxis(
                        labels_size=30,
                        title=ft.Text("TPR"),
                    ),
                    bottom_axis=ft.ChartAxis(
                        labels_size=30,
                        title=ft.Text("FPR"),
                    ),
                    tooltip_bgcolor=ft.colors.with_opacity(0.8, ft.colors.BLUE_GREY),
                    min_x=0,
                    max_x=1,
                    min_y=0,
                    max_y=1,
                    width=450,
                    height=400,
                ),
                # padding=ft.padding.only(left=20, right=10, top=10, bottom=20),
            )
            
            return chart
            
        except Exception as e:
            logger.error(f"Error creating ROC chart: {e}")
            return ft.Text(f"Error creating ROC curve: {str(e)}", color=ft.colors.RED_600)
    
    def _update_controls(self):
        """Update control button states."""
        has_selection = self.selected_session is not None
        is_running = has_selection and self.selected_session.status == 'running'
        
        # Update button states
        buttons = self.controls_container.content.controls
        for i, button in enumerate(buttons):
            if i == 0:  # Cancel Session button
                button.disabled = not has_selection or not is_running
            else:
                button.disabled = not has_selection
        
        self.controls_container.update()
    
    def _cancel_session(self, e):
        """Cancel a running session."""
        if self.selected_session and self.selected_session.status == 'running':
            session_id = self.selected_session.session_id
            if self.session_manager.cancel_session(session_id):
                logger.info(f"Cancelled session {session_id}")
                self._load_sessions()
            else:
                logger.error(f"Failed to cancel session {session_id}")
    
    def _refresh_sessions(self, e):
        """Refresh the session list."""
        self.selected_session = None
        self._load_sessions()
        logger.info("Sessions refreshed")
    
    def _delete_session(self, e):
        """Delete selected session."""
        if self.selected_session:
            session_id = self.selected_session.session_id
            if self.session_manager.delete_session(session_id):
                logger.info(f"Deleted session {session_id}")
                self.selected_session = None
                self._load_sessions()
            else:
                logger.error(f"Failed to delete session {session_id}")
    
    def _view_logs(self, e):
        """View logs for selected session."""
        if self.selected_session:
            session = self.selected_session
            logs = self.session_manager.get_session_logs(session.session_id, lines=1000)  # Load more lines
            
            if logs:
                # Create a simple log viewer dialog
                log_text = "".join(logs)
                log_dialog = ft.AlertDialog(
                    title=ft.Text(f"Logs for {session.name}"),
                    content=ft.Container(
                        content=ft.Text(log_text, selectable=True, size=10),
                        width=600,
                        height=400,
                    ),
                    actions=[
                        ft.TextButton("Close", on_click=lambda e: self.page.close(log_dialog))
                    ]
                )
                self.page.dialog = log_dialog
                log_dialog.open = True
                self.page.update()
            else:
                logger.warning(f"No logs available for session {session.session_id}")
    
    def _go_back(self, e):
        """Navigate back to main page."""
        self.page.go("/")
    
    def _go_to_main(self, e):
        """Navigate to main page (alternative method)."""
        self.page.go("/")
    
    def console_log(self, message: str, level: str = "INFO"):
        """Log a message to console (placeholder for now)."""
        print(f"[{level}] {message}")
    
    def initialize(self):
        """Initialize the monitor page after it's added to the page."""
        self._load_sessions()

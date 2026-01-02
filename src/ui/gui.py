import flet as ft
from ..data.loaders import get_dataset, get_dataset_full
from ..models.architectures import get_class
import inspect
from ..config.args import (_SETTINGS, 
                  _OPTIMIZERS, 
                  _LOSSES, 
                  autopartial, 
                  parse_defaultconfig, 
                  parse_hyperparameters_from_dict)
from ..core.callbacks import GuiCallback
import logging
from ..utils.helpers import create_auto_auc_components
import numpy as np
import io
import queue
import threading
import wandb
from .session_manager import get_session_manager
import sys
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class Parameter(ft.Column):
    def __init__(self, name, default, val, log = False, cat = False):
        super().__init__()
        self.name = name
        if val is None: val = "'None'"
        elif isinstance(val, str): val = "\'" + val + "\'"
        elif val is inspect._empty: val = "'None'"
        
        if default is None: default = "'None'"
        elif isinstance(default, str): default = "\'" + default + "\'"
        elif default is inspect._empty: default = "'None'"

        self.edit = ft.TextField(value = str(val), label = self.name, width = 220)
        
        self.low = ft.TextField(label = self.name + ' min', width = 105, visible = False)
        self.high = ft.TextField(label = 'max', width = 105, visible = False)
        if not cat:
            self.low.value = str(val[0])
            self.high.value = str(val[1])
            self.edit.visible = False
            self.low.visible = True
            self.high.visible = True

        self.default = ft.TextField(value = str(default), label = "default value", width = 100)

        # invalid input, assume value is categorical
        if cat and log: 
            log = False

        # Custom log scale switchable button
        self.log = ft.Row([
            ft.IconButton(
                icon=ft.icons.MOVING,
                tooltip="Log scale",
                on_click=self.log_toggle,
                style=ft.ButtonStyle(
                    bgcolor=ft.colors.BLUE_100 if log else ft.colors.GREY_100,
                    color=ft.colors.BLUE_700 if log else ft.colors.GREY_600,
                )
            )
        ], visible=False if cat else True)
        self.switch = ft.CupertinoSlidingSegmentedButton(
            selected_index = 1 if cat else 0,
            disabled = log,
            thumb_color = ft.colors.BLUE_400,
            on_change = self.categorical_changed,
            padding = ft.padding.symmetric(0, 10),
            controls=[
                ft.Icon(ft.icons.TRENDING_FLAT, tooltip="Interval"),
                ft.Icon(ft.icons.CATEGORY_OUTLINED, tooltip="Categorical"),
            ],
        )

        self.controls = [
            ft.Row(
                # alignment = ft.MainAxisAlignment.START,
                # vertical_alignment = ft.CrossAxisAlignment.CENTER,
                controls = [
                            self.edit,
                            self.low,
                            self.high,
                            self.default,
                            self.switch,
                            self.log,
                        ] 
            )
        ]
    
    def log_toggle(self, e):
        # Toggle the log scale state
        current_state = self.log.controls[0].style.bgcolor == ft.colors.BLUE_100
        new_state = not current_state
        
        # Update button style based on new state
        self.log.controls[0].style.bgcolor = ft.colors.BLUE_100 if new_state else ft.colors.GREY_100
        self.log.controls[0].style.color = ft.colors.BLUE_700 if new_state else ft.colors.GREY_600
        
        # Update the switch disabled state
        self.switch.disabled = new_state
        self.update()
    
    def categorical_changed(self, e):
        if self.switch.selected_index == 1:
            self.log.visible = False
            self.edit.visible = True
            self.low.visible = False
            self.high.visible = False
        else:
            self.log.visible = True
            self.edit.visible = False
            self.low.visible = True
            self.high.visible = True
        self.update()


    def export_setting(self):
        data = {}
        if self.switch.selected_index == 0:
            data["val"] = (eval(self.low.value), eval(self.high.value))
        else :
            data["val"] = eval(self.edit.value)
            
        # Check if log scale is enabled by checking button state
        log_enabled = self.log.controls[0].style.bgcolor == ft.colors.BLUE_100
        if log_enabled: 
            data["log"] = True
        if self.default.value != 'None':
            data["default"] = eval(self.default.value)
        return data

class Space(ft.Column):
    def __init__(self, name, src):
        super().__init__()
        self.width = 600
        self.alignment = ft.MainAxisAlignment.START
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.name = name
        self.title = ft.Dropdown(
            label = name,
            on_change=self.object_changed,
        )
        self.src = src
        for k in src.keys():
            self.title.options.append(ft.dropdown.Option(k))

        self.export_btn = ft.IconButton(
            icon = ft.icons.CONTENT_COPY,
            tooltip = "copy",
            on_click = self.copy_setting,
        )
        self.expand_btn = ft.IconButton(
            icon = ft.icons.ADD,
            tooltip = "more",
            on_click = self.expand_setting,
        )

        self.paras = ft.Column()
        self.controls = [
            self.title,
            self.paras,
            ft.Row(
                alignment = ft.MainAxisAlignment.CENTER,
                vertical_alignment = ft.CrossAxisAlignment.CENTER,
                controls = [
                    self.export_btn,
                    self.expand_btn
                ]
            )
        ]

    @property
    def locked(self):
        return self.title.disabled

    @locked.setter
    def locked(self, value):
        self.title.disabled = value

    def object_changed(self, e):
        conf = parse_defaultconfig(self.title.value)
        self.set_paras(conf[self.name])
        self.ref.set_paras(conf[self.ref.name])
    
    def set_paras(self, data):
        self.paras.controls = []
        self.title.value = data["type"]
        for k, v in data["space"].items():
            if isinstance(v["val"], (int, str, float)):
                para = Parameter(k, v["val"], [v["val"]], False, True)
            elif isinstance(v["val"], list):
                para = Parameter(k, v.get("default", None), v["val"], False, True)
            else:
                log = v.get("log", False)
                para = Parameter(k, v.get("default", None), v["val"], log, False)
            self.paras.controls.append(para)
        self.update()
    
    def expand_setting(self, e):
        cls = get_class(*self.src[self.title.value])
        signature = inspect.signature(cls)
        for v in signature.parameters.values():
            if not v.name in self:
                self.paras.controls.append(Parameter(v.name, v.default, [v.default], False, True))
        self.update()

    def copy_setting(self, e):
        data = self.export_setting()
        self.page.set_clipboard(str(data))
        self.page.open(ft.SnackBar(ft.Text(f"Setting Copied"), open=True))

    def export_setting(self):
        data = {}
        data["type"] = self.title.value
        space = {}
        for para in self.paras.controls:
            space[para.name] = para.export_setting()
        data["space"] = space
        return data
    
    def __contains__(self, m):
        for para in self.paras.controls:
            if para.name == m:
                return True
        return False

class AutoAUCApp(ft.Column):
    def __init__(self):
        super().__init__()
        # Use global session manager
        self.session_manager = get_session_manager()
        self.current_session = None
        self.models = ft.Dropdown(
            label = "Model Architecture",
            hint_text = "Choose the model you want to finetune",
            width = 300,
            options = [
                ft.dropdown.Option("resnet18"),
                ft.dropdown.Option("resnet20"),
                ft.dropdown.Option("resnet32"),
            ],
        )
        self.datasets = ft.Dropdown(
            label = "Dataset",
            width = 300,
            on_change=self.datasets_changed,
            options = [
                ft.dropdown.Option("CIFAR10"),
                ft.dropdown.Option("CIFAR100"),
            ],
        )
        self.inspect_btn = ft.IconButton(
            icon = ft.icons.ANALYTICS,
            tooltip = "Inspect Dataset",
            on_click = self.datasets_inspected,
        )
        self.dataset_stat = ft.Text("")
        self.progress_ring = ft.ProgressRing(visible = False)

        self.targets = ft.Dropdown(
            label = "Target Metric",
            width = 300,
            value = "Customed",
            options = [
                ft.dropdown.Option("Customed"),
            ],
            on_change=self.target_changed
        )
        for k in _SETTINGS.keys():
            self.targets.options.append(ft.dropdown.Option(k))

        self.fpr = ft.TextField(value = "0.3", label = "Max FPR", visible = False)
        self.tpr = ft.TextField(value = "0.7", label = "Min TPR", visible = False)
        self.pretrained = False
        self.pretrained_ck = ft.Checkbox(
            value=False, label="pretrained", on_change=self.pretrained_changed
        )
        self.model_path = ft.TextField(label="Model Path", visible = False)

        self.pick_files_dialog = ft.FilePicker(on_result=self.pick_files_result)
        self.browse_btn = ft.ElevatedButton(
                                "Browse",
                                on_click = lambda _: self.pick_files_dialog.pick_files(
                                    allow_multiple=True
                                ),
                                visible = False,
                                bgcolor = ft.colors.BLUE_600,
                                color = ft.colors.WHITE
                            )

        self.optimizers = Space("optimizer", _OPTIMIZERS)
        self.losses = Space("loss", _LOSSES)
        self.optimizers.ref = self.losses
        self.losses.ref = self.optimizers

        # Training parameter text fields
        self.seed = ft.TextField(value = "123", label = "SEED")
        self.batch_size = ft.TextField(value = "128", label = "Batch Size")
        self.eval_batch_size = ft.TextField(value = "128", label = "Eval Batch Size")
        self.sampling_rate = ft.TextField(value = "0.2", label = "Sampling Rate")
        self.epochs = ft.TextField(value = "50", label = "Epochs")
        self.decay_epochs = ft.TextField(value = "[0.5, 0.75]", label = "Decay Epochs")
        self.num_workers = ft.TextField(value = "0", label = "Num Workers")
        self.output_path = ft.TextField(value = "'./output'", label = "Output Path")

        self.n_trials = ft.TextField(value = "5", label = "N Trials")
        self.n_configs = ft.TextField(value = "1", label = "N Configs")
        
        self.advanced = ft.ExpansionTile(
            title = ft.Text("Advanced Settings", size=16, weight=ft.FontWeight.BOLD),
            affinity = ft.TileAffinity.PLATFORM,
            maintain_state = True,
            controls = [
                ft.Container(
                    alignment=ft.alignment.center,
                    height=20,
                ),
                ft.Container(
                    content=ft.Row(
                        alignment = ft.MainAxisAlignment.START,
                        vertical_alignment = ft.CrossAxisAlignment.START,
                        spacing = 30,
                        controls = [
                            self.optimizers,
                            self.losses
                        ]
                    ),
                    padding = ft.padding.only(left=20),
                ),
                ft.Container(
                    alignment=ft.alignment.center,
                    height=20,
                )
            ]
        )

        self.others = ft.ExpansionTile(
            title = ft.Text("Training Parameters", size=16, weight=ft.FontWeight.BOLD),
            affinity = ft.TileAffinity.PLATFORM,
            maintain_state = True,
            controls = [
                ft.Container(
                    alignment=ft.alignment.center,
                    height=20,
                ),
                ft.Container(
                    content=ft.ResponsiveRow([
                        ft.Column(col={"sm": 3}, controls=[self.seed]),
                        ft.Column(col={"sm": 3}, controls=[self.batch_size]),
                        ft.Column(col={"sm": 3}, controls=[self.eval_batch_size]),
                        ft.Column(col={"sm": 3}, controls=[self.sampling_rate]),
                    ]),
                    padding = ft.padding.only(left=20),
                ),
                ft.Container(
                    alignment=ft.alignment.center,
                    height=20,
                ),
                ft.Container(
                    content=ft.ResponsiveRow([
                        ft.Column(col={"sm": 3}, controls=[self.epochs]),
                        ft.Column(col={"sm": 3}, controls=[self.decay_epochs]),
                        ft.Column(col={"sm": 3}, controls=[self.num_workers]),
                        ft.Column(col={"sm": 3}, controls=[self.output_path]),
                    ]),
                    padding = ft.padding.only(left=20),
                ),
                ft.Container(
                    alignment=ft.alignment.center,
                    height=20,
                ),
                ft.Container(
                    content=ft.ResponsiveRow([
                        ft.Column(col={"sm": 3}, controls=[self.n_trials]),
                        ft.Column(col={"sm": 3}, controls=[self.n_configs])
                    ]),
                    padding = ft.padding.only(left=20),
                ),
                ft.Container(
                    alignment=ft.alignment.center,
                    height=20,
                ),
            ]
        )
        
        self.train_btn = ft.ElevatedButton(
            "Start Training",
            on_click = self.train,
            bgcolor = ft.colors.BLUE_600,
            color = ft.colors.WHITE
        )
        self.cancel_btn = ft.ElevatedButton(
            "Cancel Training",
            on_click = self.cancel_training,
            visible = False,
            bgcolor = ft.colors.RED_600,
            color = ft.colors.WHITE
        )
        self.monitor_btn = ft.ElevatedButton(
            "Monitor Sessions",
            on_click = self.go_to_monitor,
            bgcolor = ft.colors.GREEN_600,
            color = ft.colors.WHITE,
            icon = ft.icons.MONITOR_HEART
        )
        self.train_pb = ft.ProgressBar(visible = False, width = 400)
        self.is_training = False
        self.training_thread = None
        self.training_process = None

        # Create a better console with fixed size and proper scrolling
        self.console_text = ft.Text("", selectable=True, size=12, font_family="Consolas, 'Courier New', monospace")
        self.console_scroll = ft.ListView(
            [self.console_text],
            height=300,
            width=1200,
            spacing=2,
            padding=ft.padding.all(10),
            auto_scroll=True,
        )
        
        # Clear console button
        self.clear_console_btn = ft.IconButton(
            icon=ft.icons.CLEAR,
            tooltip="Clear Console",
            on_click=self.clear_console,
            bgcolor=ft.colors.RED_100,
            icon_color=ft.colors.RED_700,
        )
        
        # Console header with clear button
        self.console_header = ft.Row(
            [
                ft.Text("Training Console", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.BOLD),
                self.clear_console_btn,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )
        
        # Console container with better styling
        self.console = ft.Container(
            content=ft.Column([
                self.console_header,
                ft.Container(
                    content=self.console_scroll,
                    width=1200,
                    height=300,
                    bgcolor=ft.colors.GREY_50,
                    border=ft.border.all(2, ft.colors.GREY_300),
                    border_radius=8,
                    padding=ft.padding.all(0),
                )
            ]),
            width=1200,
            height=350,
        )
        
        # Initialize console text (welcome messages will be added after page is created)
        self.console_text.value = ""
        
        # Simple, clean header
        self.header = ft.Container(
            content=ft.Text("AutoMAX - Automated AUC Optimization", 
                          size=24, weight=ft.FontWeight.BOLD, color=ft.colors.GREY_800),
            alignment=ft.alignment.center,
            padding=ft.padding.all(15),
            margin=ft.margin.only(bottom=15),
        )
        
        self.controls = [
            self.header,
            # Model Selection Section
            ft.Container(
                content=ft.Column([
                    ft.Text("Model Configuration", size=16, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        alignment = ft.MainAxisAlignment.START,
                        vertical_alignment = ft.CrossAxisAlignment.CENTER,
                        controls = [
                            self.models,
                            self.pretrained_ck, 
                            self.model_path, 
                            self.browse_btn
                        ] 
                    ),
                ], spacing=10),
                border=ft.border.all(1, ft.colors.GREY_300),
                border_radius=ft.border_radius.all(4),
                padding=ft.padding.all(15),
                margin=ft.margin.only(bottom=15),
            ),
            # Dataset Selection Section
            ft.Container(
                content=ft.Column([
                    ft.Text("Dataset Configuration", size=16, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        alignment = ft.MainAxisAlignment.START,
                        vertical_alignment = ft.CrossAxisAlignment.CENTER,
                        controls = [
                            self.datasets,
                            self.progress_ring,
                            self.inspect_btn,
                            self.dataset_stat,
                        ]
                    ),
                ], spacing=10),
                border=ft.border.all(1, ft.colors.GREY_300),
                border_radius=ft.border_radius.all(4),
                padding=ft.padding.all(15),
                margin=ft.margin.only(bottom=15),
            ),
            # Target Selection Section
            ft.Container(
                content=ft.Column([
                    ft.Text("Target Configuration", size=16, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        alignment = ft.MainAxisAlignment.START,
                        vertical_alignment = ft.CrossAxisAlignment.CENTER,
                        controls = [
                            self.targets,
                            self.fpr,
                            self.tpr
                        ]
                    ),
                ], spacing=10),
                border=ft.border.all(1, ft.colors.GREY_300),
                border_radius=ft.border_radius.all(4),
                padding=ft.padding.all(15),
                margin=ft.margin.only(bottom=15),
            ),
            self.advanced,
            self.others,
            # Training Control Section
            ft.Container(
                content=ft.Column([
                    ft.Text("Training Control", size=16, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        alignment = ft.MainAxisAlignment.CENTER,
                        controls = [
                            self.train_btn,
                            self.cancel_btn,
                            self.monitor_btn,
                        ]
                    ),
                    self.train_pb,
                ], spacing=15),
                border=ft.border.all(1, ft.colors.GREY_300),
                border_radius=ft.border_radius.all(4),
                padding=ft.padding.all(15),
                margin=ft.margin.only(bottom=15),
            ),
            self.console,
            self.pick_files_dialog
        ]

    def pick_files_result(self, e: ft.FilePickerResultEvent):
        self.model_path.value = (
            ", ".join(map(lambda f: f.path, e.files)) if e.files else ""
        )
        print(self.model_path.value)
        self.model_path.update()

    def pretrained_changed(self, e):
        self.pretrained = self.pretrained_ck.value
        self.model_path.visible = self.pretrained
        self.browse_btn.visible = self.pretrained
        self.update()
    
    def target_changed(self, e):
        if self.targets.value == 'Customed':
            # self.optimizers.locked = False
            # self.losses.locked = False
            for option in self.optimizers.title.options:
                option.visible = True
            for option in self.losses.title.options:
                option.visible = True
        else:
            #self.optimizers.locked = True
            #self.losses.locked = True
            for option in self.optimizers.title.options:
                option.visible = False
            for option in self.losses.title.options:
                option.visible = False
            flag = False
            for setting in _SETTINGS[self.targets.value]:
                conf = parse_defaultconfig(setting)
                for option in self.optimizers.title.options:
                    if (option.key == conf["optimizer"]["type"]):
                        option.visible = True
                for option in self.losses.title.options:
                    if (option.key == conf["loss"]["type"]):
                        option.visible = True
                if not flag:
                    self.optimizers.set_paras(conf["optimizer"])
                    self.losses.set_paras(conf["loss"])
                    flag = True
                
        self.optimizers.update()
        self.losses.update()

        if self.targets.value == 'OPAUC' or self.targets.value == 'TPAUC':
            self.fpr.visible = True
        else:
            self.fpr.visible = False
        
        if self.targets.value == 'TPAUC':
            self.tpr.visible = True
        else:
            self.tpr.visible = False
        
        self.tpr.update()
        self.fpr.update()
    
    def datasets_inspected(self, e):
        if self.datasets.value is None:
            self.dataset_stat.value = "Please choose a dataset"
            self.dataset_stat.update()
        else:
            self.inspect_btn.visible = False
            self.progress_ring.visible = True
            self.update()
            try:
                trainset, validset = get_dataset(self.datasets.value)
            except:
                self.dataset_stat.value = "Error Occured"
            else:
                len1 = len(trainset)
                len2 = len(validset)
                self.dataset_stat.value = f"Training set size: {len1}   Validation set size: {len2}"
            finally:
                self.inspect_btn.visible = True
                self.progress_ring.visible = False
            self.update()
    
    def datasets_changed(self, e):
        self.dataset_stat.value = ""
        self.dataset_stat.update()
    
    def set_progressbar(self, value):
        self.train_pb.value = value
        try:
            self.train_pb.update()
        except Exception:
            # Control not yet connected to page, skip update
            pass
    
    def console_log(self, s, level="INFO"):
        # Add timestamp to log messages
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Color coding for different log levels
        level_colors = {
            "INFO": "",
            "WARNING": "⚠️ ",
            "ERROR": "❌ ",
            "SUCCESS": "✅ ",
            "TRAINING": "🚀 "
        }
        
        level_prefix = level_colors.get(level, "")
        log_message = f"[{timestamp}] {level_prefix}{s}"
        
        # Update console text
        if self.console_text.value:
            self.console_text.value += '\n' + log_message
        else:
            self.console_text.value = log_message
        
        # Only update if the control is properly connected to a page
        try:
            self.console_text.update()
            # Auto-scroll to bottom
            self.console_scroll.scroll_to(offset=-1, duration=100)
        except Exception:
            # Control not yet connected to page, skip update
            pass
        
    def text_pass(self, s):
        # Use the same improved logging
        self.console_log(s)
    
    def clear_console(self, e):
        """Clear the console output."""
        self.console_text.value = ""
        self.console_text.update()
        self.console_log("Console cleared")
    
    def initialize(self):
        """Initialize the console after it's added to the page."""
        self.console_log("AutoAUC Training Console Ready", "SUCCESS")
        self.console_log("Configure your training parameters and click 'Start Training' to begin", "INFO")

    def train(self, e):
        self.train_btn.disabled = True
        self.cancel_btn.visible = True
        self.train_pb.visible = True
        self.update()
        
        # Create a new training session
        session_name = f"Training_{self.models.value}_{self.datasets.value}_{self.targets.value}"
        config = {
            'model': self.models.value,
            'dataset': self.datasets.value,
            'target': self.targets.value,
            'epochs': eval(self.epochs.value),
            'batch_size': eval(self.batch_size.value),
            'optimizer': self.optimizers.export_setting()["type"],
            'loss': self.losses.export_setting()["type"],
            'output_path': eval(self.output_path.value)
        }
        
        self.current_session = self.session_manager.create_session(session_name, config).session_id
        self.console_log(f"Created training session: {self.current_session}", "SUCCESS")
        
        # Register GUI callback with session manager for progress updates
        self.session_manager.register_gui_callback(self, self.current_session)
        
        # Start training in a separate process
        self._start_training_process()

        self.is_training = True

    def _add_type_indicators(self, space_dict):
        """Add type indicators to distinguish between categorical and interval hyperparameters."""
        processed = {}
        for key, value in space_dict.items():
            if isinstance(value, dict) and 'val' in value:
                # This is a hyperparameter with the full structure
                val = value['val']
                processed_param = {
                    'default': value.get('default'),
                    'log': value.get('log', False)
                }
                
                if isinstance(val, tuple) and len(val) == 2:
                    # It's an interval (continuous range)
                    processed_param['type'] = 'interval'
                    processed_param['val'] = list(val)  # Convert to list for JSON serialization
                elif isinstance(val, list):
                    # It's categorical
                    processed_param['type'] = 'categorical'
                    processed_param['val'] = val
                else:
                    # Single value
                    processed_param['type'] = 'single'
                    processed_param['val'] = val
                
                processed[key] = processed_param
            else:
                # Direct value (fallback for simple cases)
                processed[key] = value
        return processed

    def _start_training_process(self):
        """Start training in a separate process."""
        import subprocess
        import tempfile
        import json
        
        # Create temporary config file
        config = {
            'model': self.models.value,
            'dataset': self.datasets.value,
            'target': self.targets.value,
            'epochs': eval(self.epochs.value),
            'batch_size': eval(self.batch_size.value),
            'eval_batch_size': eval(self.eval_batch_size.value),
            'sampling_rate': eval(self.sampling_rate.value),
            'decay_epochs': eval(self.decay_epochs.value),
            'num_workers': eval(self.num_workers.value),
            'output_path': eval(self.output_path.value),
            'seed': eval(self.seed.value),
            'n_trials': eval(self.n_trials.value),
            'n_configs': eval(self.n_configs.value),
            'pretrained': self.pretrained,
            'model_path': self.model_path.value if self.pretrained else None,
            'optimizer': self.optimizers.export_setting()["type"],
            'loss': self.losses.export_setting()["type"],
            'optimizer_space': self._add_type_indicators(self.optimizers.export_setting()["space"]),
            'loss_space': self._add_type_indicators(self.losses.export_setting()["space"])
        }
        
        # Add target-specific parameters
        if self.targets.value == 'OPAUC' or self.targets.value == 'TPAUC':
            config['max_fpr'] = float(self.fpr.value)
        if self.targets.value == 'TPAUC':
            config['min_tpr'] = float(self.tpr.value)
        
        # Save config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_file = f.name
        
        # Start the training process
        try:
            process = subprocess.Popen([
                sys.executable, 
                str(Path(__file__).parent / "training_worker.py"),
                "--config", config_file,
                "--session-id", self.current_session
            ])
            
            self.console_log(f"Started training process (PID: {process.pid})", "SUCCESS")
            self.console_log("Training is running independently. You can close the GUI if needed.", "INFO")
            
            # Store process reference for monitoring
            self.training_process = process
            
        except Exception as e:
            self.console_log(f"Failed to start training process: {str(e)}", "ERROR")
            # Clean up config file
            os.unlink(config_file)
            self._training_finished()


    def cancel_training(self, e):
        cur_pid = self.session_manager.get_session(self.current_session).pid
        if cur_pid == os.getpid():
            self.console_log("Training is initializing, cannot be cancelled", "WARNING")
            return
            
        if self.is_training:
            self.is_training = False
            self.console_log("Training cancelled by user", "WARNING")
            
            # Cancel the session
            if self.current_session:
                self.session_manager.cancel_session(self.current_session)
                self.session_manager.write_session_log(
                    self.current_session,
                    "Training cancelled by user",
                    "WARNING"
                )
            
            # The session manager will handle killing the process
            # since we now store the actual training process PID
            
            self._training_finished()

    def _training_finished(self):
        self.train_btn.disabled = False
        self.cancel_btn.visible = False
        self.train_pb.visible = False
        self.train_pb.value = 0
        self.is_training = False
        
        # Unregister GUI callback
        if self.current_session:
            self.session_manager.unregister_gui_callback()
        
        self.current_session = None
        self.training_process = None
        self.update()

    def update_chart(self, metrics):
        # Log metrics to wandb
        try:
            wandb.log(metrics)
        except Exception:
            # Wandb not available or error, continue without logging
            pass
    
    def go_to_monitor(self, e):
        """Navigate to the monitor page."""
        self.page.go("/monitor")

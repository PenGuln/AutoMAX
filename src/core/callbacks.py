"""
Training callbacks for AutoAUC framework.
"""

import logging
from typing import Any, Dict, List, Optional

from ..config.args import TrainingArguments, AutoAUCConfigration

logger = logging.getLogger(__name__)


class TrainerState:
    """State object to track training progress."""
    
    def __init__(self):
        self.epoch = 0
        self.total_epoch = 0
        self.step = 0


class TrainerCallback:
    """
    Base callback class for training events.
    
    All callback methods are optional and can be overridden in subclasses.
    """
    
    def __init__(self) -> None:
        pass
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of trainer initialization."""
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of training."""
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of training."""
        pass

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of an epoch."""
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of an epoch."""
        pass

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of a training step."""
        pass

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of a substep during gradient accumulation."""
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of a training step."""
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after an evaluation phase."""
        pass

    def on_predict(self, args: TrainingArguments, state: TrainerState, metrics, **kwargs):
        """Event called after a successful prediction."""
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after a checkpoint save."""
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after logging the last logs."""
        pass

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after a prediction step."""
        pass


class CallbackHandler(TrainerCallback):
    """
    Handler that manages multiple callbacks and calls them in order.
    """
    
    def __init__(self, callbacks: List[TrainerCallback], model, optimizer, loss_fn):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = None
        self.eval_dataloader = None

    def add_callback(self, callback):
        """Add a callback to the handler."""
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. "
                f"The current list of callbacks is:\n{self.callback_list}"
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        """Remove and return a callback."""
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        """Remove a callback without returning it."""
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        """Get a string representation of all callbacks."""
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_init_end", args, state)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_train_begin", args, state)

    def on_train_end(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_train_end", args, state)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_epoch_begin", args, state)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, metrics, **kwargs):
        return self._call_event("on_epoch_end", args, state, metrics=metrics, **kwargs)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_step_begin", args, state)

    def on_substep_end(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_substep_end", args, state)

    def on_step_end(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_step_end", args, state)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_evaluate", args, state)

    def on_predict(self, args: TrainingArguments, state: TrainerState, metrics):
        return self._call_event("on_predict", args, state, metrics=metrics)

    def on_save(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_save", args, state)

    def on_log(self, args: TrainingArguments, state: TrainerState, logs):
        return self._call_event("on_log", args, state, logs=logs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_prediction_step", args, state)

    def _call_event(self, event, args, state, **kwargs):
        """Call the specified event on all callbacks."""
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                model=self.model,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )


class DefaultCallback(TrainerCallback):
    """Default callback with basic functionality."""
    
    def __init__(self, auto_config: AutoAUCConfigration) -> None:
        self.tot_trials = auto_config.n_trials
        self.cur_trial = 0
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of training."""
        self.cur_trial += 1

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of an epoch."""
        optimizer = kwargs.get("optimizer", None)
        if optimizer and state.epoch in args.decay_epochs:
            optimizer.update_regularizer(decay_factor=10)  # decrease learning rate by 10x & update regularizer

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of an epoch."""
        state.epoch += 1

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of a training step."""
        state.step += 1


class CLICallback(TrainerCallback):
    """Callback for command-line interface with detailed logging and optional wandb."""
    
    def __init__(self, auto_config: AutoAUCConfigration, wandb_project: Optional[str] = None,
                 wandb_experiment: str = "experiment") -> None:
        self.tot_trials = auto_config.n_trials
        self.cur_trial = 0
        self.best_metric = float('-inf')
        self._wandb_project = wandb_project
        self._wandb_experiment = wandb_experiment
        self._use_wandb = wandb_project is not None
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of training."""
        if self._use_wandb:
            try:
                import wandb
                run_name = f"{self._wandb_experiment}_{self.cur_trial + 1}"
                wandb.init(project=self._wandb_project, name=run_name, reinit=True)
            except ImportError:
                logger.warning("wandb not installed; skipping wandb logging")
                self._use_wandb = False
        print(f"\nStarting training round {self.cur_trial + 1}/{self.tot_trials}")
        print(f"Epochs: {state.total_epoch}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {kwargs['optimizer'].lr}")
        print("-" * 50)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of an epoch."""
        optimizer = kwargs.get("optimizer", None)
        if optimizer and state.epoch in args.decay_epochs:
            optimizer.update_regularizer(decay_factor=10)  # decrease learning rate by 10x & update regularizer
            print(f"\nLearning rate decayed to: {optimizer.lr}")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of an epoch."""
        metrics = kwargs.get("metrics", {})
        if self._use_wandb:
            try:
                import wandb
                wandb.log(metrics)
            except ImportError:
                pass
        # Get the main metric value (excluding 'target' and 'epoch')
        statement = f"Epoch {state.epoch + 1}/{state.total_epoch} | Loss: {metrics.get('loss', 0):.4f} | "
        for k, v in metrics.items():
            if k not in ['epoch', 'lr', 'loss']:
                # Only format numeric values, skip dicts/lists/arrays
                if isinstance(v, (int, float)) or (hasattr(v, '__float__') and not isinstance(v, (dict, list, tuple))):
                    try:
                        statement += f"{k}: {float(v):.4f} | "
                    except (ValueError, TypeError):
                        pass  # Skip non-numeric values
        statement += f"LR: {metrics.get('lr', 0):.6f}"
        print(statement)
        
        state.epoch += 1

    def on_train_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of training."""
        if self._use_wandb:
            try:
                import wandb
                wandb.finish()
            except ImportError:
                pass
        self.cur_trial += 1
        print("\n" + "=" * 50)
        print(f"Training round {self.cur_trial}/{self.tot_trials} completed")
        print("=" * 50 + "\n")
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of a training step."""
        state.step += 1


class SessionCallback(TrainerCallback):
    """Callback for session management with progress updates and logging."""
    
    def __init__(self, session_manager, session_id: str, auto_config: AutoAUCConfigration) -> None:
        self.session_manager = session_manager
        self.session_id = session_id
        self.tot_trials = auto_config.n_trials
        self.cur_trial = 0
        self._last_update_time = 0
        self._cached_session = None
        self._cached_time = 0
        self._cache_ttl = 10  # Cache for 10 seconds
    
    def _get_cached_session(self):
        """Get cached session info to avoid repeated file I/O."""
        import time
        current_time = time.time()
        if (self._cached_session is None or 
            current_time - self._cached_time > self._cache_ttl):
            self._cached_session = self.session_manager.get_session(self.session_id)
            self._cached_time = current_time
        return self._cached_session
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of trainer initialization."""
        self.session_manager.write_session_log(self.session_id, f"Trial {self.cur_trial} Initialized")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of training."""
        self.session_manager.write_session_log(self.session_id, f'Trial {self.cur_trial} Start Training')
        # Update session progress
        progress = self.cur_trial / self.tot_trials
        self.session_manager.update_session(self.session_id, progress=progress)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of training."""
        self.cur_trial += 1

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of an epoch."""
        # Check if session is still active with caching
        session = self._get_cached_session()
        if session and session.status == 'cancelled':
            raise KeyboardInterrupt("Training cancelled by user")
            
        optimizer = kwargs.get("optimizer", None)
        if optimizer and state.epoch in args.decay_epochs:
            optimizer.update_regularizer(decay_factor=10)  # decrease learning rate by 10x & update regularizer

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of an epoch."""
        import time
        import numpy as np
        current_time = time.time()
        
        # Check if session is still active with caching
        session = self._get_cached_session()
        if session and session.status == 'cancelled':
            raise KeyboardInterrupt("Training cancelled by user")
        
        metrics = kwargs.get("metrics", {})
        current_loss = metrics.get('loss', 0.0)
        
        # Get test_true and test_pred from kwargs (from last epoch)
        test_true = kwargs.get("test_true")
        test_pred = kwargs.get("test_pred")
        
        # Convert numpy arrays to lists for storage
        test_true_list = None
        test_pred_list = None
        if test_true is not None:
            if isinstance(test_true, np.ndarray):
                test_true_list = test_true.tolist()
            else:
                test_true_list = list(test_true)
        if test_pred is not None:
            if isinstance(test_pred, np.ndarray):
                test_pred_list = test_pred.tolist()
            else:
                test_pred_list = list(test_pred)
        
        # Update session progress and loss, and save test_true/test_pred
        progress = (state.epoch / state.total_epoch + self.cur_trial) / self.tot_trials
        update_kwargs = {
            'current_epoch': state.epoch + 1,
            'progress': progress,
            'loss': current_loss
        }
        if test_true_list is not None:
            update_kwargs['test_true'] = test_true_list
        if test_pred_list is not None:
            update_kwargs['test_pred'] = test_pred_list
        
        self.session_manager.update_session(
            self.session_id, 
            **update_kwargs
        )
        self._last_update_time = current_time
        
        # Get the main metric value (excluding 'target' and 'epoch')
        statement = f"Epoch {state.epoch + 1}/{state.total_epoch} | Loss: {current_loss:.4f} | "
        for k, v in metrics.items():
            if k not in ['epoch', 'lr', 'loss']:
                # Only format numeric values, skip dicts/lists/arrays
                if isinstance(v, (int, float)) or (hasattr(v, '__float__') and not isinstance(v, (dict, list, tuple))):
                    try:
                        statement += f"{k}: {float(v):.4f} | "
                    except (ValueError, TypeError):
                        pass  # Skip non-numeric values
        statement += f"LR: {metrics.get('lr', 0):.6f}"
        self.session_manager.write_session_log(self.session_id, statement)
        
        state.epoch += 1

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of a training step."""
        # Check if session is still active with caching
        session = self._get_cached_session()
        if session and session.status == 'cancelled':
            raise KeyboardInterrupt("Training cancelled by user")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of a training step."""
        state.step += 1

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after an evaluation phase."""
        pass

    def on_predict(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after a successful prediction."""
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after a checkpoint save."""
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after logging the last logs."""
        pass

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after a prediction step."""
        pass


class GuiCallback(TrainerCallback):
    """Callback for GUI interface with progress updates and wandb logging."""
    
    def __init__(self, app, auto_config: AutoAUCConfigration) -> None:
        self.app = app
        self.tot_trials = auto_config.n_trials
        self.cur_trial = 0
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of trainer initialization."""
        self.app.console_log(f"Trial {self.cur_trial} Initialized")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of training."""
        self.app.console_log(f'Trial {self.cur_trial} Start Training')
        self.app.set_progressbar((state.epoch / state.total_epoch + self.cur_trial) / self.tot_trials)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of training."""
        self.cur_trial += 1

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of an epoch."""
        if not self.app.is_training:
            raise KeyboardInterrupt("Training cancelled by user")
            
        optimizer = kwargs.get("optimizer", None)
        if optimizer and state.epoch in args.decay_epochs:
            optimizer.update_regularizer(decay_factor=10)  # decrease learning rate by 10x & update regularizer

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of an epoch."""
        if not self.app.is_training:
            raise KeyboardInterrupt("Training cancelled by user")
            
        self.app.set_progressbar((state.epoch / state.total_epoch + self.cur_trial) / self.tot_trials)
        metrics = kwargs.get("metrics", {})
        
        # Get the main metric value (excluding 'target' and 'epoch')
        statement = f"Epoch {state.epoch + 1}/{state.total_epoch} | Loss: {metrics.get('loss', 0):.4f} | "
        for k, v in metrics.items():
            if k not in ['epoch', 'lr', 'loss']:
                # Only format numeric values, skip dicts/lists/arrays
                if isinstance(v, (int, float)) or (hasattr(v, '__float__') and not isinstance(v, (dict, list, tuple))):
                    try:
                        statement += f"{k}: {float(v):.4f} | "
                    except (ValueError, TypeError):
                        pass  # Skip non-numeric values
        statement += f"LR: {metrics.get('lr', 0):.6f}"
        self.app.console_log(statement)
        
        # Log metrics to wandb
        self.app.update_chart(metrics)
        

        state.epoch += 1

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of a training step."""
        if not self.app.is_training:
            raise KeyboardInterrupt("Training cancelled by user")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of a training step."""
        state.step += 1

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after an evaluation phase."""
        pass

    def on_predict(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after a successful prediction."""
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after a checkpoint save."""
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after logging the last logs."""
        pass

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after a prediction step."""
        pass

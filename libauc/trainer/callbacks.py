"""
Training callbacks for AutoAUC framework.
"""

import logging
from typing import Any, Dict, List, Optional

from .args import TrainingArguments

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
                **kwargs,
            )


class DefaultCallback(TrainerCallback):
    """Default callback with basic functionality."""
    
    def __init__(self) -> None:
        super().__init__()
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of an epoch."""
        optimizer = kwargs.get("optimizer")
        if optimizer and state.epoch in args.decay_epochs:
            if getattr(optimizer, "model_ref", None) is not None:
                optimizer.update_regularizer(decay_factor=10)
            else:
                optimizer.update_lr(decay_factor=10)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of an epoch."""
        state.epoch += 1

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of a training step."""
        state.step += 1


class CLICallback(TrainerCallback):
    """Callback for command-line interface with detailed logging."""
    
    def __init__(self) -> None:
        super().__init__()
        self._use_wandb = True
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of training."""
        try:
            import wandb
            wandb.init(project=args.project_name, name=args.experiment_name, reinit=True)
        except ImportError:
            logger.warning("wandb not installed; skipping wandb logging")
            self._use_wandb = False

        optimizer = kwargs.get("optimizer")
        lr_str = f"{optimizer.lr:.6f}" if optimizer else "N/A"
        print(f"Epochs:        {state.total_epoch}")
        print(f"Batch size:    {args.batch_size}")
        print(f"Learning rate: {lr_str}")
        print(f"Loss:          {args.loss}")
        print(f"Optimizer:     {args.optimizer}")
        print("-" * 50)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of an epoch."""
        optimizer = kwargs.get("optimizer")
        if optimizer and state.epoch in args.decay_epochs:
            if getattr(optimizer, "model_ref", None) is not None:
                optimizer.update_regularizer(decay_factor=10)
            else:
                optimizer.update_lr(decay_factor=10)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of an epoch."""
        metrics:    list  = kwargs.get("metrics", [])
        train_loss: float = kwargs.get("train_loss", 0)
        lr:         float = kwargs.get("lr", 0)

        # -- Build the flat log dict (used for both wandb and console) ----
        log: dict[str, float] = {
            "epoch":      state.epoch + 1,
            "train_loss": train_loss,
            "lr":         lr,
        }

        single = len(metrics) == 1
        first_metric_val = None

        for ds_idx, ds_metrics in enumerate(metrics):
            if not isinstance(ds_metrics, dict):
                continue
            prefix = "" if single else f"ds{ds_idx + 1}/"
            for k, v in ds_metrics.items():
                if k in ("epoch", "lr", "loss"):
                    continue
                try:
                    fval = float(v)
                    log[f"{prefix}{k}"] = fval
                    if first_metric_val is None:
                        first_metric_val = fval
                except (ValueError, TypeError):
                    pass

        # -- Console output -----------------------------------------------
        display_parts = [f"Epoch {state.epoch + 1}/{state.total_epoch}",
                         f"Loss: {train_loss:.4f}"]
        for k, v in log.items():
            if k in ("epoch", "train_loss", "lr"):
                continue
            display_parts.append(f"{k}: {v:.4f}")
        display_parts.append(f"LR: {lr:.6f}")
        print(" | ".join(display_parts))

        # -- wandb logging ------------------------------------------------
        if self._use_wandb:
            try:
                import wandb
                wandb.log(log, step=state.epoch + 1)
            except Exception as e:
                logger.warning(f"wandb logging failed: {e}")

        state.epoch += 1

    def on_train_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of training."""
        print("-" * 50)
        print(f"Training complete.")
        if self._use_wandb:
            try:
                import wandb
                wandb.finish()
            except ImportError:
                pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of a training step."""
        state.step += 1
"""Explicit recoverable inference errors; messages are never sent to CSA logs."""
class InferenceUnavailable(RuntimeError):
    """A transient inference resource failure, not a model/programming fault."""

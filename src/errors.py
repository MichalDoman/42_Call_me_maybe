class CallMeMaybeError(Exception):
    """Base calss for map custom errors."""

    def __init__(self, message: str) -> None:
        """Add custom error prefix to the message."""
        message_with_prefix: str = f"{self.__class__.__name__}: {message}"
        super().__init__(message_with_prefix)


class ParsingError(CallMeMaybeError):
    """Custom error used during map parsing."""

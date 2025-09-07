class PDFConversionError(Exception):
    """Base exception for all PDF conversion errors."""

    pass


class UnexpectedStopError(PDFConversionError):
    """Exception raised when the conversion process stops unexpectedly."""

    pass


class EmptyConversionError(PDFConversionError):
    """Exception raised when the conversion result is empty."""

    pass

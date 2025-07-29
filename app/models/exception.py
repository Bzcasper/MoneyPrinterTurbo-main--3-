import traceback
from typing import Any

from loguru import logger


class HttpException(Exception):
    def __init__(
        self, task_id: str, status_code: int, message: str = "", data: Any = None
    ):
        self.message = message
        self.status_code = status_code
        self.data = data
        # Retrieve the exception stack trace information.
        tb_str = traceback.format_exc().strip()
        if not tb_str or tb_str == "NoneType: None":
            msg = f"HttpException: {status_code}, {task_id}, {message}"
        else:
            msg = f"HttpException: {status_code}, {task_id}, {message}\n{tb_str}"

        if status_code == 400:
            logger.warning(msg)
        else:
            logger.error(msg)


class FileNotFoundException(Exception):
    pass


class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors."""
    pass


class SupabaseConnectionError(Exception):
    """
    Custom exception for Supabase connection-related errors.
    
    This exception is raised when there are issues specifically related to
    Supabase connections, authentication, or service availability.
    """
    
    def __init__(self, message: str, original_error: Exception = None):
        """
        Initialize SupabaseConnectionError.
        
        Args:
            message: Error message describing the issue
            original_error: Optional original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.original_error = original_error
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.original_error:
            return f"{self.message} (Original error: {str(self.original_error)})"
        return self.message

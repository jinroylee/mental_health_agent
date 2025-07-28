"""
pretty logging utility for the mental health assistant.
"""

import logging
from typing import Any


class Colors:
    """ANSI color codes for terminal output."""
    # Basic colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Special formatting
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Reset
    RESET = '\033[0m'


class prettyLogger:
    """Enhanced logger with pretty output for different log types."""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
    
    def node_separator_top(self, node_name: str) -> None:
        """Print pretty node separator."""
        separator = f"#{node_name}#".center(50, '⬇')
        pretty_separator = f"{Colors.BLUE}{Colors.BOLD}{separator}{Colors.RESET}"
        print(pretty_separator)
        # self.logger.info(separator)
    
    def node_separator_bottom(self, node_name: str) -> None:
        """Print pretty node separator."""
        separator = f"#{node_name}#".center(50, '⬆')
        pretty_separator = f"{Colors.BLUE}{Colors.BOLD}{separator}{Colors.RESET}"
        print(pretty_separator)
        # self.logger.info(separator)
    
    def function_separator(self, text: str) -> None:
        """Print pretty function separator."""
        separator = f"=== {text} ==="
        pretty_separator = f"{Colors.YELLOW}{Colors.BOLD}{separator}{Colors.RESET}"
        print(pretty_separator)
    
    def logger_separator_top(self, text: str) -> None:
        """Print pretty logger separator."""
        separator = f"={'=' * 10}{text}{'↓' * 18}"
        pretty_separator = f"{Colors.CYAN}{Colors.BOLD}{separator}{Colors.RESET}"
        print(pretty_separator)
        # self.logger.info(separator)
    
    def logger_separator_bottom(self, text: str) -> None:
        """Print pretty logger separator."""
        separator = f"={'=' * 10}{text}{'↑' * 18}"
        pretty_separator = f"{Colors.CYAN}{Colors.BOLD}{separator}{Colors.RESET}"
        print(pretty_separator)
        # self.logger.info(separator)
    
    def state_print(self, key: str, value: Any) -> None:
        """Print pretty state/variable information."""
        pretty_output = f"{Colors.BRIGHT_GREEN}{key}:{Colors.RESET} {Colors.WHITE}{value}{Colors.RESET}"
        print(pretty_output)
    
    def info(self, message: str, *args) -> None:
        """pretty info log."""
        pretty_message = f"{Colors.GREEN}INFO:{Colors.RESET} {message}"
        print(pretty_message % args if args else pretty_message)
        # self.logger.info(message, *args)
    
    def warning(self, message: str, *args) -> None:
        """pretty warning log."""
        pretty_message = f"{Colors.YELLOW}WARNING:{Colors.RESET} {message}"
        print(pretty_message % args if args else pretty_message)
        # self.logger.warning(message, *args)
    
    def error(self, message: str, *args) -> None:
        """pretty error log."""
        pretty_message = f"{Colors.RED}ERROR:{Colors.RESET} {message}"
        print(pretty_message % args if args else pretty_message)
        # self.logger.error(message, *args)
    
    def debug(self, message: str, *args) -> None:
        """pretty debug log."""
        pretty_message = f"{Colors.DIM}DEBUG:{Colors.RESET} {message}"
        print(pretty_message % args if args else pretty_message)
        self.logger.debug(message, *args)
    
    def success(self, message: str) -> None:
        """Print pretty success message."""
        pretty_message = f"{Colors.BRIGHT_GREEN}✓ {message}{Colors.RESET}"
        print(pretty_message)
    
    def content_block(self, content: str, title: str = None) -> None:
        """Print pretty content block."""
        if title:
            title_line = f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}{title}{Colors.RESET}"
            print(title_line)
        
        # Split content into lines and add pretty prefix
        lines = content.split('\n')
        for line in lines:
            pretty_line = f"{Colors.MAGENTA}│{Colors.RESET} {line}"
            print(pretty_line)
    
    def separator_line(self, char: str = "─", length: int = 50) -> None:
        """Print a pretty separator line."""
        line = char * length
        pretty_line = f"{Colors.DIM}{line}{Colors.RESET}"
        print(pretty_line)


def get_pretty_logger(name: str) -> prettyLogger:
    """Get a pretty logger instance."""
    return prettyLogger(name) 
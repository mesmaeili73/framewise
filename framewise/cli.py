"""Command-line interface for FrameWise.

This module provides a simple CLI for FrameWise, currently supporting
version information and help commands. Future versions may add commands
for processing videos directly from the command line.

Example:
    Command line usage::

        $ framewise version
        FrameWise version 0.1.2
        
        $ framewise help
        Usage: framewise <command> [options]
        ...
"""

from __future__ import annotations

import sys
from pathlib import Path
from loguru import logger


def main() -> None:
    """Main CLI entry point.
    
    Parses command-line arguments and dispatches to appropriate handlers.
    Currently supports 'version' and 'help' commands.
    
    Commands:
        version: Display the current FrameWise version
        help: Show usage information
    
    Example:
        >>> # Called when user runs: framewise version
        >>> main()  # with sys.argv = ['framewise', 'version']
    """
    logger.info("🎬 FrameWise - AI-powered video tutorial assistant")
    logger.info("=" * 50)
    
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1]
    
    if command == "version":
        from framewise import __version__
        print(f"FrameWise version {__version__}")
    elif command == "help":
        print_usage()
    else:
        logger.error(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


def print_usage() -> None:
    """Print CLI usage information.
    
    Displays available commands and basic usage instructions.
    Directs users to the online documentation for detailed information.
    
    Example:
        >>> print_usage()
        Usage: framewise <command> [options]
        
        Commands:
            version     Show version information
            help        Show this help message
        ...
    """
    usage = """
Usage: framewise <command> [options]

Commands:
    version     Show version information
    help        Show this help message

For detailed usage, please refer to the documentation:
https://github.com/mesmaeili73/framewise#readme

Quick Start:
    from framewise import TranscriptExtractor, FrameExtractor
    
    # Extract transcript
    extractor = TranscriptExtractor()
    transcript = extractor.extract("video.mp4")
    
    # Extract frames
    frame_ext = FrameExtractor()
    frames = frame_ext.extract("video.mp4", transcript=transcript)
"""
    print(usage)


if __name__ == "__main__":
    main()

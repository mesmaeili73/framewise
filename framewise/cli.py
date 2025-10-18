"""
Command-line interface for FrameWise
"""

import sys
from pathlib import Path
from loguru import logger


def main():
    """Main CLI entry point"""
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


def print_usage():
    """Print usage information"""
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

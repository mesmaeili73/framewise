"""
FrameWise: AI-powered video tutorial assistant

Transform tutorial videos into instant, visual support with intelligent
frame extraction and multimodal RAG.
"""

__version__ = "0.1.0"

from framewise.core import (
    TranscriptExtractor,
    Transcript,
    TranscriptSegment,
    FrameExtractor,
    ExtractedFrame,
)
from framewise.embeddings import FrameWiseEmbedder
from framewise.retrieval import FrameWiseVectorStore, FrameWiseQA

__all__ = [
    "TranscriptExtractor",
    "Transcript",
    "TranscriptSegment",
    "FrameExtractor",
    "ExtractedFrame",
    "FrameWiseEmbedder",
    "FrameWiseVectorStore",
    "FrameWiseQA",
]

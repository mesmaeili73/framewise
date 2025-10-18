"""
FrameWise: AI-powered video tutorial assistant

Transform tutorial videos into instant, visual support with intelligent
frame extraction and multimodal RAG.
"""

__version__ = "0.1.2"

from framewise.core import (
    TranscriptExtractor,
    Transcript,
    TranscriptSegment,
    FrameExtractor,
    ExtractedFrame,
)
from framewise.embeddings import FrameWiseEmbedder
from framewise.retrieval import FrameWiseVectorStore

# Optional LLM features
try:
    from framewise.retrieval import FrameWiseQA
    _has_llm = True
except ImportError:
    _has_llm = False
    FrameWiseQA = None

__all__ = [
    "TranscriptExtractor",
    "Transcript",
    "TranscriptSegment",
    "FrameExtractor",
    "ExtractedFrame",
    "FrameWiseEmbedder",
    "FrameWiseVectorStore",
]

if _has_llm:
    __all__.append("FrameWiseQA")

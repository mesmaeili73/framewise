"""
Vector search and retrieval functionality
"""

from framewise.retrieval.vector_store import FrameWiseVectorStore

# Optional Q&A system (requires LLM dependencies)
try:
    from framewise.retrieval.qa_system import FrameWiseQA
    _has_qa = True
except ImportError:
    _has_qa = False
    FrameWiseQA = None

__all__ = ["FrameWiseVectorStore"]

if _has_qa:
    __all__.append("FrameWiseQA")

"""
FrameWise: AI-powered video tutorial assistant

Transform tutorial videos into instant, visual support with intelligent
frame extraction and multimodal RAG.
"""

__version__ = "0.1.0"

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 6e8959a (feat: Add transcript extraction with Whisper)
from framewise.core import (
    TranscriptExtractor,
    Transcript,
    TranscriptSegment,
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 4338149 (feat: Add intelligent frame extraction with scene detection)
    FrameExtractor,
    ExtractedFrame,
)
from framewise.embeddings import FrameWiseEmbedder
<<<<<<< HEAD
<<<<<<< HEAD
from framewise.retrieval import FrameWiseVectorStore, FrameWiseQA
=======
)
>>>>>>> 6e8959a (feat: Add transcript extraction with Whisper)
=======
from framewise.retrieval import FrameWiseVectorStore
>>>>>>> 3dda9dd (feat: Add multimodal embeddings and vector search (Phase 3))
=======
from framewise.retrieval import FrameWiseVectorStore, FrameWiseQA
>>>>>>> 25bbc40 (feat: Add transcript correction and complete LLM integration)

__all__ = [
    "TranscriptExtractor",
    "Transcript",
    "TranscriptSegment",
<<<<<<< HEAD
<<<<<<< HEAD
    "FrameExtractor",
    "ExtractedFrame",
    "FrameWiseEmbedder",
    "FrameWiseVectorStore",
<<<<<<< HEAD
<<<<<<< HEAD
    "FrameWiseQA",
=======
    "FrameExtractor",
    "ExtractedFrame",
>>>>>>> 4338149 (feat: Add intelligent frame extraction with scene detection)
=======
>>>>>>> 3dda9dd (feat: Add multimodal embeddings and vector search (Phase 3))
=======
    "FrameWiseQA",
>>>>>>> 25bbc40 (feat: Add transcript correction and complete LLM integration)
]
=======
# Core imports will be added as we build the modules
__all__ = []
>>>>>>> db3361e (Initial commit: Project structure and Poetry setup)
=======
]
>>>>>>> 6e8959a (feat: Add transcript extraction with Whisper)

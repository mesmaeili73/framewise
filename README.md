# 🎬 FrameWise

*AI-powered video search and Q&A for tutorial content*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform tutorial videos into searchable, interactive content. Find exact moments by meaning, not just keywords.

## Quick Start

```bash
pip install framewise

# Or install with LLM features
pip install framewise[llm]
```

### Basic Usage

```python
from framewise import TranscriptExtractor, FrameExtractor, FrameWiseEmbedder, FrameWiseVectorStore

# 1. Process video
transcript = TranscriptExtractor().extract("tutorial.mp4")
frames = FrameExtractor().extract("tutorial.mp4", transcript)

# 2. Create searchable index
embedder = FrameWiseEmbedder()
embeddings = embedder.embed_frames_batch(frames)

store = FrameWiseVectorStore()
store.create_table(embeddings)

# 3. Search
results = store.search_by_text("How do I export?", embedder, limit=3)
for r in results:
    print(f"{r['timestamp']}s: {r['text']}")
```

### With LLM Q&A (Optional)

```python
from framewise import FrameWiseQA

# Requires: pip install framewise[llm]
# Set ANTHROPIC_API_KEY environment variable

qa = FrameWiseQA(vector_store=store, embedder=embedder)
response = qa.ask("How do I get started?")
print(response['answer'])  # Natural language answer with frame references
```

## Features

- 🎙️ **Transcript Extraction** - Whisper-powered speech-to-text
- 🖼️ **Smart Frame Extraction** - Captures key visual moments
- 🧠 **Multimodal Embeddings** - CLIP + Sentence Transformers
- 🔍 **Semantic Search** - Find by meaning, not keywords
- 🤖 **LLM Q&A** - Optional Claude integration (requires API key)

## How It Works

```
Video → Transcript (Whisper) → Frames (OpenCV) → Embeddings (CLIP) → Search (LanceDB) → [Optional] Q&A (Claude)
```

**Core Pipeline** (no API keys needed):
1. Extract audio transcript with timestamps
2. Capture key frames at important moments
3. Generate multimodal embeddings
4. Search by semantic similarity

**Optional LLM Layer**:
- Add natural language Q&A with Claude
- Requires `pip install framewise[llm]` and API key

## Installation

### Requirements
- Python 3.9+
- ffmpeg (for video processing)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
apt-get install ffmpeg
```

### Install FrameWise

```bash
# Core features only
pip install framewise

# With LLM Q&A support
pip install framewise[llm]

# From source
git clone https://github.com/mesmaeili73/framewise.git
cd framewise
pip install -e .
```

## Configuration

### Frame Extraction

```python
extractor = FrameExtractor(
    strategy="hybrid",        # "scene", "transcript", or "hybrid"
    max_frames_per_video=20,
    scene_threshold=0.3,
    quality_threshold=0.5
)
```

### Embeddings

```python
embedder = FrameWiseEmbedder(
    text_model="all-MiniLM-L6-v2",
    vision_model="openai/clip-vit-base-patch32",
    device="cuda"  # or "cpu"
)
```

### LLM Q&A (Optional)

```python
# Set environment variable
export ANTHROPIC_API_KEY=your_key_here

# Or pass directly
qa = FrameWiseQA(
    vector_store=store,
    embedder=embedder,
    model="claude-3-5-sonnet-20241022",
    api_key="your_key_here"
)
```

## Current Limitations (V1)

- **Vector Store**: LanceDB only (local storage)
- **LLM Provider**: Claude/Anthropic only
- **Embedding Models**: Fixed (CLIP + Sentence Transformers)

Future versions will support multiple vector stores (Qdrant, Elasticsearch), LLM providers (OpenAI, VertexAI), and configurable embedding models.

## Examples

See the [`examples/`](examples/) directory for complete examples:
- `extract_transcript.py` - Basic transcript extraction
- `extract_frames.py` - Frame extraction strategies
- `complete_pipeline.py` - Full end-to-end workflow

## Use Cases

- **Product Teams**: Build AI assistants for tutorial libraries
- **EdTech**: Make educational videos searchable
- **Documentation**: Create interactive video knowledge bases

## Performance

For 50 videos (5 min each):
- **Processing**: ~15-90 min (GPU vs CPU)
- **Search**: <50ms per query
- **Q&A**: ~2-3 seconds (with LLM)

## Contributing

Contributions welcome! This is an open-source project.

## License

MIT License - see [LICENSE](LICENSE) file

## Built With

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [CLIP](https://github.com/openai/CLIP) - Vision-language embeddings
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [LanceDB](https://lancedb.com/) - Vector database
- [LangChain](https://www.langchain.com/) - LLM orchestration
- [Anthropic Claude](https://www.anthropic.com/) - Language model

---

*FrameWise: See the right frame at the right time* 🎬

# 🎬 FrameWise

*Your AI guide that knows exactly which frame matters*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is FrameWise?

FrameWise is an intelligent virtual assistant that transforms tutorial videos into instant, visual support. Instead of watching entire videos, users get the exact screenshot and explanation they need—right when they need it.

**The Problem:** You have 50+ tutorial videos (5 min each) showing users how to use your products. Users don't want to watch full videos to find one answer.

**The Solution:** FrameWise combines AI-powered video analysis, multimodal embeddings, and LLM integration to provide instant, accurate answers with visual proof.

## ✨ Features

- 🎙️ **Transcript Extraction** - Automatic speech-to-text with Whisper
- 🖼️ **Smart Frame Extraction** - Intelligent keyframe selection at important moments
- 🧠 **Multimodal Embeddings** - CLIP for images + sentence-transformers for text
- 🔍 **Semantic Search** - Find relevant content by meaning, not just keywords
- 🤖 **LLM Q&A** - Natural language answers with Claude (via LangChain)
- 🔧 **Transcript Correction** - Fix common speech recognition errors
- ⚡ **GPU Acceleration** - Fast processing with CUDA support
- 📦 **Batch Processing** - Handle multiple videos efficiently

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/framewise.git
cd framewise

# Install dependencies
pip install -e .

# Or with Poetry
poetry install
poetry shell
```

### Setup

1. **Install ffmpeg** (required for video processing):
```bash
brew install ffmpeg  # macOS
# or
apt-get install ffmpeg  # Linux
```

2. **Configure API keys** (for LLM features):
```bash
cp .env.example .env
# Edit .env and add your Claude API key
```

### Basic Usage

```python
from framewise import (
    TranscriptExtractor,
    FrameExtractor,
    FrameWiseEmbedder,
    FrameWiseVectorStore,
    FrameWiseQA,
)

# 1. Extract transcript
transcript_ext = TranscriptExtractor()
transcript = transcript_ext.extract("tutorial.mp4")

# 2. Extract keyframes
frame_ext = FrameExtractor(strategy="hybrid")
frames = frame_ext.extract("tutorial.mp4", transcript=transcript)

# 3. Generate embeddings
embedder = FrameWiseEmbedder()
embeddings = embedder.embed_frames_batch(frames)

# 4. Store in vector database
store = FrameWiseVectorStore()
store.create_table(embeddings)

# 5. Ask questions!
qa = FrameWiseQA(vector_store=store, embedder=embedder)
response = qa.ask("How do I export my data?")

print(response['answer'])
# "To export your data, click the Export button in the top-right corner..."

print(f"See frame at {response['relevant_frames'][0]['timestamp']}s")
# Direct link to the relevant moment
```

## 📖 How It Works

### The Pipeline

```
Video Input
    ↓
1. Transcript Extraction (Whisper)
    ↓
2. Frame Extraction (OpenCV + Smart Alignment)
    ↓
3. Multimodal Embeddings (CLIP + Sentence Transformers)
    ↓
4. Vector Database (LanceDB)
    ↓
5. Semantic Search + LLM Q&A (Claude)
    ↓
Natural Language Answer + Visual Proof
```

### Intelligent Frame Extraction

FrameWise uses three strategies to extract the most relevant frames:

- **Scene Detection**: Captures visual transitions and UI changes
- **Transcript Alignment**: Extracts frames when action keywords are mentioned ("click", "select", "open")
- **Hybrid** (Recommended): Combines both for optimal coverage

### Multimodal Search

Unlike traditional search that only matches keywords, FrameWise understands **meaning**:

- Query: "export data" → Finds: "Click the export button" ✅
- Query: "save file" → Finds: "Click save icon" ✅
- Query: "settings" → Finds frames showing settings UI ✅

## 🎯 Use Cases

### For Product Teams
- Build an AI assistant for your tutorial library
- Reduce support tickets with instant visual answers
- Scale documentation without writing more docs

### For EdTech Platforms
- Make educational videos searchable
- Provide instant answers to student questions
- Improve learning outcomes with visual guidance

### For Documentation Teams
- Augment written docs with video content
- Create interactive video knowledge bases
- Enable natural language search across videos

## 📊 Performance

For 50 tutorial videos (5 min each):

| Task | CPU | GPU (Single) | GPU (Multi) |
|------|-----|--------------|-------------|
| Transcript Extraction | ~50 min | ~15 min | ~8 min |
| Frame Extraction | ~25 min | ~10 min | ~5 min |
| Embedding Generation | ~15 min | ~3 min | ~1.5 min |
| **Total Processing** | **~90 min** | **~28 min** | **~15 min** |

**Search Speed**: <50ms per query  
**Answer Generation**: ~2-3 seconds (with Claude)

## 🧪 Examples

### Example 1: Simple Search

```python
from framewise import FrameWiseVectorStore, FrameWiseEmbedder

store = FrameWiseVectorStore(db_path="tutorials.db")
embedder = FrameWiseEmbedder()

results = store.search_by_text(
    "How do I export?",
    embedder=embedder,
    limit=3
)

for result in results:
    print(f"{result['timestamp']}s: {result['text']}")
```

### Example 2: Q&A with Claude

```python
from framewise import FrameWiseQA

qa = FrameWiseQA(vector_store=store, embedder=embedder)
response = qa.ask("How do I get started?")

print(response['answer'])
# Natural language answer from Claude

for frame in response['relevant_frames']:
    print(f"See: {frame['frame_path']} at {frame['timestamp']}s")
```

### Example 3: Batch Processing

```python
from framewise import TranscriptExtractor, FrameExtractor

# Process multiple videos
videos = ["tutorial1.mp4", "tutorial2.mp4", "tutorial3.mp4"]

transcript_ext = TranscriptExtractor()
transcripts = transcript_ext.extract_batch(videos, output_dir="transcripts/")

# Extract frames from all
frame_ext = FrameExtractor(strategy="hybrid")
for video, transcript in zip(videos, transcripts):
    frames = frame_ext.extract(video, transcript=transcript)
```

### Example 4: Transcript Correction

```python
from framewise.utils.transcript_corrections import TranscriptCorrector

# Fix common transcription errors
corrector = TranscriptCorrector({
    "Defali": "Definely",
    "expot": "export",
})

corrected_transcript = corrector.correct_transcript(transcript)
```

## 🛠️ Configuration

### Frame Extraction Settings

```python
frame_extractor = FrameExtractor(
    strategy="hybrid",           # "scene", "transcript", or "hybrid"
    max_frames_per_video=20,     # Limit number of frames
    scene_threshold=0.3,         # Scene change sensitivity (0-1)
    quality_threshold=0.5        # Minimum frame quality (0-1)
)
```

### Embedding Models

```python
embedder = FrameWiseEmbedder(
    text_model="all-MiniLM-L6-v2",              # Fast, good quality
    vision_model="openai/clip-vit-base-patch32", # Balanced
    device="cuda"                                # Use GPU
)
```

### LLM Configuration

```python
qa = FrameWiseQA(
    vector_store=store,
    embedder=embedder,
    model="claude-3-5-sonnet-20241022",  # Claude model
    max_tokens=1024,                      # Response length
    temperature=0.7                       # Creativity (0-1)
)
```

## 📁 Project Structure

```
framewise/
├── framewise/              # Main package
│   ├── core/              # Video processing
│   │   ├── transcript_extractor.py
│   │   └── frame_extractor.py
│   ├── embeddings/        # Embedding generation
│   │   └── embedder.py
│   ├── retrieval/         # Vector search & Q&A
│   │   ├── vector_store.py
│   │   └── qa_system.py
│   └── utils/             # Utilities
│       └── transcript_corrections.py
├── examples/              # Usage examples
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Test transcript extraction
python test_it_yourself.py

# Test vector search
python test_search.py

# Test Q&A with Claude
python test_qa.py

# Test with Tableau video
python test_tableau.py
```

## 🤝 Contributing

Contributions are welcome! This is an open-source project.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with:
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [CLIP](https://github.com/openai/CLIP) - Vision-language model
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [LanceDB](https://lancedb.com/) - Vector database
- [LangChain](https://www.langchain.com/) - LLM orchestration
- [Anthropic Claude](https://www.anthropic.com/) - Language model

## 🎯 Why "FrameWise"?

Because wisdom isn't about watching everything—it's about seeing the right frame at the right time.

---

*Built for product teams who want their users to succeed, one frame at a time.* 🎬✨

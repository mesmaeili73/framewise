# TranscriptExtractor - Complete Explanation

## What Does It Do?

The `TranscriptExtractor` converts **video audio** into **text transcripts** with precise timestamps. Think of it as automatic subtitles/captions generation.

## How It Works

### 1. **Input**: A video file
```python
video_file = "tutorial.mp4"  # Any video format
```

### 2. **Processing**: Whisper AI analyzes the audio
- Extracts audio from video
- Uses OpenAI's Whisper model (speech-to-text AI)
- Detects language automatically (or you can specify)
- Breaks transcript into segments with timestamps

### 3. **Output**: Structured transcript data

```json
{
  "video_path": "tutorial.mp4",
  "language": "en",
  "full_text": "Welcome to this tutorial. Today we'll learn about exports...",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Welcome to this tutorial"
    },
    {
      "start": 2.5,
      "end": 5.0,
      "text": "Today we'll learn about exports"
    },
    {
      "start": 5.0,
      "end": 8.0,
      "text": "Click the export button here"
    }
  ]
}
```

## Why Is This Useful?

### For FrameWise:
1. **Searchable Content**: Convert videos into searchable text
2. **Timestamp Alignment**: Know exactly when something was said
3. **Frame Extraction**: Use timestamps to extract frames at key moments
4. **LLM Context**: Feed transcripts to LLM for answering questions

### Example Use Case:
```
User asks: "How do I export data?"

1. Search transcript for "export"
2. Find: "Click the export button here" at 5.0s
3. Extract frame at 5.0s showing the export button
4. LLM uses transcript + frame to answer
```

## Key Features

### 1. **Multiple Model Sizes**
```python
# Faster but less accurate
extractor = TranscriptExtractor(model_size="tiny")

# Balanced (default)
extractor = TranscriptExtractor(model_size="base")

# Most accurate but slower
extractor = TranscriptExtractor(model_size="large")
```

### 2. **Language Support**
```python
# Auto-detect language
extractor = TranscriptExtractor()

# Force specific language
extractor = TranscriptExtractor(language="en")  # English
extractor = TranscriptExtractor(language="es")  # Spanish
```

### 3. **GPU Acceleration**
```python
# Use GPU for faster processing
extractor = TranscriptExtractor(device="cuda")

# Use CPU
extractor = TranscriptExtractor(device="cpu")
```

### 4. **Batch Processing**
```python
# Process multiple videos at once
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
transcripts = extractor.extract_batch(videos, output_dir="transcripts/")
```

## Data Structure Explained

### TranscriptSegment
A single piece of spoken text with timing:
```python
segment = TranscriptSegment(
    start=2.5,      # Starts at 2.5 seconds
    end=5.0,        # Ends at 5.0 seconds
    text="Hello"    # What was said
)
```

### Transcript
Complete transcript with all segments:
```python
transcript = Transcript(
    video_path=Path("video.mp4"),
    language="en",
    segments=[segment1, segment2, ...],
    full_text="Complete transcript text..."
)
```

## Real-World Example

### Input Video (5 minutes):
A tutorial showing how to use a software application

### What Happens:
1. **Audio Extraction**: Whisper extracts audio from video
2. **Speech Recognition**: AI converts speech to text
3. **Segmentation**: Breaks into ~30 segments (one every 10 seconds)
4. **Timestamps**: Each segment has precise start/end times

### Output:
```python
transcript.language  # "en"
transcript.segments  # 30 segments
transcript.full_text # "Welcome to this tutorial. First, open the app..."

# Access specific segment
segment = transcript.segments[5]
print(f"At {segment.start}s: {segment.text}")
# Output: "At 25.0s: Click the settings button"
```

## How FrameWise Will Use This

```
Step 1: Extract Transcript (DONE ✅)
   ↓
   "At 25.0s: Click the settings button"
   
Step 2: Extract Frame at 25.0s (NEXT)
   ↓
   [Screenshot showing settings button]
   
Step 3: Create Embeddings (FUTURE)
   ↓
   Vector: [0.23, 0.45, 0.12, ...]
   
Step 4: User Asks Question (FUTURE)
   ↓
   "Where is the settings button?"
   
Step 5: Retrieve + Answer (FUTURE)
   ↓
   LLM: "The settings button is in the top-right corner"
   + Shows screenshot from 25.0s
```

## Performance

For a 5-minute video:
- **tiny model**: ~30 seconds (CPU)
- **base model**: ~1 minute (CPU), ~15 seconds (GPU)
- **large model**: ~3 minutes (CPU), ~30 seconds (GPU)

For 50 videos (5 min each):
- **Sequential**: ~50 minutes (base model, CPU)
- **Batch + GPU**: ~12 minutes (base model, 2 GPUs)

## Limitations

1. **Requires audio**: Won't work on silent videos
2. **Accuracy**: Depends on audio quality and accents
3. **Language**: Works best with English, but supports 90+ languages
4. **Processing time**: Large models are slow on CPU

## Next Steps

After transcript extraction, we'll add:
1. **Frame extraction** at key timestamps
2. **Embedding generation** for semantic search
3. **Vector database** for fast retrieval
4. **LLM integration** for answering questions

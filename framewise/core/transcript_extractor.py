"""Transcript extraction from video files using OpenAI Whisper.

This module provides functionality to extract audio transcripts from video files
using the Whisper speech recognition model. It supports multiple languages,
different model sizes, and batch processing.

Example:
    Basic usage::

        from framewise import TranscriptExtractor
        
        extractor = TranscriptExtractor(model_size="base")
        transcript = extractor.extract("video.mp4")
        
        print(f"Language: {transcript.language}")
        print(f"Text: {transcript.full_text}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
import json


@dataclass
class TranscriptSegment:
    """A segment of transcribed text with timing information.
    
    Represents a single segment of transcribed speech with start and end timestamps.
    Segments are typically sentence or phrase-level chunks of the transcript.
    
    Attributes:
        start: Start time of the segment in seconds.
        end: End time of the segment in seconds.
        text: The transcribed text content for this segment.
    
    Example:
        >>> segment = TranscriptSegment(start=0.0, end=2.5, text="Hello world")
        >>> print(f"[{segment.start}s - {segment.end}s]: {segment.text}")
        [0.0s - 2.5s]: Hello world
    """
    
    start: float
    end: float
    text: str
    
    def to_dict(self) -> Dict[str, Union[float, str]]:
        """Convert segment to dictionary format.
        
        Returns:
            Dictionary containing start, end, and text fields.
            
        Example:
            >>> segment = TranscriptSegment(0.0, 2.5, "Hello")
            >>> segment.to_dict()
            {'start': 0.0, 'end': 2.5, 'text': 'Hello'}
        """
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text
        }


@dataclass
class Transcript:
    """Complete transcript with metadata and segments.
    
    Represents a full video transcript including the source video path,
    detected language, individual segments with timestamps, and the
    complete transcribed text.
    
    Attributes:
        video_path: Path to the source video file.
        language: Detected or specified language code (e.g., 'en', 'es').
        segments: List of transcript segments with timing information.
        full_text: Complete transcribed text without timestamps.
    
    Example:
        >>> transcript = Transcript(
        ...     video_path=Path("video.mp4"),
        ...     language="en",
        ...     segments=[TranscriptSegment(0.0, 2.5, "Hello world")],
        ...     full_text="Hello world"
        ... )
        >>> transcript.save("transcript.json")
    """
    
    video_path: Path
    language: str
    segments: List[TranscriptSegment]
    full_text: str
    
    def to_dict(self) -> Dict[str, Union[str, List[Dict]]]:
        """Convert transcript to dictionary format.
        
        Returns:
            Dictionary containing all transcript data including video path,
            language, segments, and full text.
            
        Example:
            >>> transcript.to_dict()
            {
                'video_path': 'video.mp4',
                'language': 'en',
                'segments': [{'start': 0.0, 'end': 2.5, 'text': 'Hello'}],
                'full_text': 'Hello'
            }
        """
        return {
            "video_path": str(self.video_path),
            "language": self.language,
            "segments": [seg.to_dict() for seg in self.segments],
            "full_text": self.full_text
        }
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save transcript to JSON file.
        
        Saves the complete transcript including all segments and metadata
        to a JSON file with UTF-8 encoding.
        
        Args:
            output_path: Path where the JSON file should be saved.
            
        Raises:
            IOError: If the file cannot be written.
            
        Example:
            >>> transcript.save("output/transcript.json")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> Transcript:
        """Load transcript from JSON file.
        
        Reads a previously saved transcript from a JSON file and reconstructs
        the Transcript object with all segments and metadata.
        
        Args:
            path: Path to the JSON file to load.
            
        Returns:
            Reconstructed Transcript object.
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If required fields are missing from the JSON.
            
        Example:
            >>> transcript = Transcript.load("transcript.json")
            >>> print(transcript.language)
            en
        """
        path = Path(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = [
            TranscriptSegment(**seg) for seg in data['segments']
        ]
        
        return cls(
            video_path=Path(data['video_path']),
            language=data['language'],
            segments=segments,
            full_text=data['full_text']
        )


class TranscriptExtractor:
    """Extract transcripts from video files using OpenAI Whisper.
    
    This class provides an interface to the Whisper speech recognition model
    for extracting transcripts from video files. It supports multiple model
    sizes, languages, and can process videos in batch.
    
    The Whisper model is loaded lazily on first use to avoid unnecessary
    initialization overhead.
    
    Attributes:
        model_size: The Whisper model size being used.
        device: The device (CPU/CUDA) the model runs on.
        language: The language code for transcription, or None for auto-detection.
    
    Example:
        Basic usage::
        
            extractor = TranscriptExtractor(model_size="base", language="en")
            transcript = extractor.extract("video.mp4")
            print(transcript.full_text)
        
        Batch processing::
        
            videos = ["video1.mp4", "video2.mp4"]
            transcripts = extractor.extract_batch(videos, output_dir="transcripts/")
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = None
    ) -> None:
        """Initialize the transcript extractor.
        
        Args:
            model_size: Whisper model size to use. Options are:
                - 'tiny': Fastest, least accurate (~1GB RAM)
                - 'base': Good balance of speed and accuracy (~1GB RAM)
                - 'small': Better accuracy (~2GB RAM)
                - 'medium': High accuracy (~5GB RAM)
                - 'large': Best accuracy (~10GB RAM)
                Defaults to 'base'.
            device: Device to run the model on. Options are:
                - 'cuda': Use GPU (requires CUDA)
                - 'cpu': Use CPU only
                - None: Auto-detect (use GPU if available)
                Defaults to None (auto-detect).
            language: ISO 639-1 language code (e.g., 'en', 'es', 'fr') or None
                for automatic language detection. Defaults to None.
        
        Example:
            >>> # Use small model with English language
            >>> extractor = TranscriptExtractor(
            ...     model_size="small",
            ...     language="en"
            ... )
            
            >>> # Use GPU if available
            >>> extractor = TranscriptExtractor(
            ...     model_size="base",
            ...     device="cuda"
            ... )
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self._model = None
    
    def _load_model(self) -> None:
        """Lazy load the Whisper model.
        
        Loads the Whisper model on first use to avoid initialization overhead
        when the extractor is created but not immediately used.
        
        Raises:
            ImportError: If openai-whisper package is not installed.
            RuntimeError: If the model fails to load.
        """
        if self._model is None:
            try:
                import whisper
                self._model = whisper.load_model(
                    self.model_size,
                    device=self.device
                )
            except ImportError:
                raise ImportError(
                    "openai-whisper is not installed. "
                    "Install it with: pip install openai-whisper"
                )
    
    def extract(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Transcript:
        """Extract transcript from a video file.
        
        Processes the video file through Whisper to extract the audio transcript
        with timestamps. The model is loaded on first use if not already loaded.
        
        Args:
            video_path: Path to the video file to transcribe. Supports common
                video formats (mp4, avi, mov, mkv, etc.).
            output_path: Optional path to save the transcript as JSON. If provided,
                the transcript will be automatically saved after extraction.
                Defaults to None (no automatic save).
        
        Returns:
            Transcript object containing the full text, language, and timestamped
            segments.
        
        Raises:
            FileNotFoundError: If the video file doesn't exist.
            RuntimeError: If Whisper fails to process the video.
            ImportError: If openai-whisper is not installed.
        
        Example:
            >>> extractor = TranscriptExtractor()
            >>> transcript = extractor.extract(
            ...     "tutorial.mp4",
            ...     output_path="transcript.json"
            ... )
            >>> print(f"Detected language: {transcript.language}")
            Detected language: en
            >>> print(f"Duration: {transcript.segments[-1].end}s")
            Duration: 125.5s
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Load model if not already loaded
        self._load_model()
        
        # Transcribe the video
        result = self._model.transcribe(
            str(video_path),
            language=self.language,
            verbose=False
        )
        
        # Convert segments
        segments = [
            TranscriptSegment(
                start=seg['start'],
                end=seg['end'],
                text=seg['text'].strip()
            )
            for seg in result['segments']
        ]
        
        # Create transcript object
        transcript = Transcript(
            video_path=video_path,
            language=result['language'],
            segments=segments,
            full_text=result['text'].strip()
        )
        
        # Save if output path provided
        if output_path:
            transcript.save(Path(output_path))
        
        return transcript
    
    def extract_batch(
        self,
        video_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Transcript]:
        """Extract transcripts from multiple videos.
        
        Processes multiple video files sequentially, extracting transcripts
        from each. Optionally saves each transcript to a JSON file in the
        specified output directory.
        
        Args:
            video_paths: List of paths to video files to transcribe.
            output_dir: Optional directory to save transcript JSON files.
                If provided, each transcript will be saved as
                "{video_stem}_transcript.json". The directory will be created
                if it doesn't exist. Defaults to None (no automatic save).
        
        Returns:
            List of Transcript objects, one for each input video in the same order.
        
        Raises:
            FileNotFoundError: If any video file doesn't exist.
            RuntimeError: If Whisper fails to process any video.
        
        Example:
            >>> extractor = TranscriptExtractor(model_size="small")
            >>> videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
            >>> transcripts = extractor.extract_batch(
            ...     videos,
            ...     output_dir="transcripts/"
            ... )
            >>> for t in transcripts:
            ...     print(f"{t.video_path.name}: {t.language}")
            video1.mp4: en
            video2.mp4: es
            video3.mp4: fr
        """
        transcripts = []
        
        for video_path in video_paths:
            video_path = Path(video_path)
            
            # Determine output path if directory provided
            output_path = None
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{video_path.stem}_transcript.json"
            
            # Extract transcript
            transcript = self.extract(video_path, output_path)
            transcripts.append(transcript)
        
        return transcripts

"""
Transcript extraction from video files using Whisper
"""

from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
import json


@dataclass
class TranscriptSegment:
    """A segment of transcribed text with timing information"""
    
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str     # Transcribed text
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text
        }


@dataclass
class Transcript:
    """Complete transcript with metadata"""
    
    video_path: Path
    language: str
    segments: List[TranscriptSegment]
    full_text: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "video_path": str(self.video_path),
            "language": self.language,
            "segments": [seg.to_dict() for seg in self.segments],
            "full_text": self.full_text
        }
    
    def save(self, output_path: Path):
        """Save transcript to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> 'Transcript':
        """Load transcript from JSON file"""
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
    """Extract transcripts from video files using Whisper"""
    
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = None
    ):
        """
        Initialize the transcript extractor
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cuda, cpu, or None for auto)
            language: Language code (e.g., 'en', 'es') or None for auto-detect
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self._model = None
    
    def _load_model(self):
        """Lazy load the Whisper model"""
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
                    "Install it with: poetry add openai-whisper"
                )
    
    def extract(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None
    ) -> Transcript:
        """
        Extract transcript from a video file
        
        Args:
            video_path: Path to the video file
            output_path: Optional path to save the transcript JSON
            
        Returns:
            Transcript object with segments and full text
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
        video_paths: List[str | Path],
        output_dir: Optional[str | Path] = None
    ) -> List[Transcript]:
        """
        Extract transcripts from multiple videos
        
        Args:
            video_paths: List of video file paths
            output_dir: Optional directory to save transcript JSONs
            
        Returns:
            List of Transcript objects
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

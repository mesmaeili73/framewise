"""
Comprehensive tests for TranscriptExtractor
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from framewise.core.transcript_extractor import (
    TranscriptExtractor,
    Transcript,
    TranscriptSegment,
)


# Fixtures

@pytest.fixture
def sample_segments():
    """Sample transcript segments for testing"""
    return [
        {"start": 0.0, "end": 2.5, "text": "Welcome to this tutorial"},
        {"start": 2.5, "end": 5.0, "text": "Today we'll learn about exports"},
        {"start": 5.0, "end": 8.0, "text": "Click the export button here"},
    ]


@pytest.fixture
def sample_whisper_result(sample_segments):
    """Mock Whisper API result"""
    return {
        "text": "Welcome to this tutorial Today we'll learn about exports Click the export button here",
        "segments": sample_segments,
        "language": "en"
    }


@pytest.fixture
def sample_transcript(tmp_path, sample_segments):
    """Sample Transcript object"""
    video_path = tmp_path / "test_video.mp4"
    segments = [TranscriptSegment(**seg) for seg in sample_segments]
    return Transcript(
        video_path=video_path,
        language="en",
        segments=segments,
        full_text="Welcome to this tutorial Today we'll learn about exports Click the export button here"
    )


@pytest.fixture
def mock_whisper_model(sample_whisper_result):
    """Mock Whisper model"""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = sample_whisper_result
    return mock_model


# Tests for TranscriptSegment

class TestTranscriptSegment:
    """Tests for TranscriptSegment dataclass"""
    
    def test_create_segment(self):
        """Test creating a transcript segment"""
        segment = TranscriptSegment(
            start=0.0,
            end=2.5,
            text="Hello world"
        )
        assert segment.start == 0.0
        assert segment.end == 2.5
        assert segment.text == "Hello world"
    
    def test_segment_to_dict(self):
        """Test converting segment to dictionary"""
        segment = TranscriptSegment(start=1.0, end=3.0, text="Test")
        result = segment.to_dict()
        
        assert result == {
            "start": 1.0,
            "end": 3.0,
            "text": "Test"
        }
    
    def test_segment_duration(self):
        """Test calculating segment duration"""
        segment = TranscriptSegment(start=1.5, end=4.5, text="Test")
        duration = segment.end - segment.start
        assert duration == 3.0


# Tests for Transcript

class TestTranscript:
    """Tests for Transcript dataclass"""
    
    def test_create_transcript(self, tmp_path, sample_segments):
        """Test creating a transcript"""
        video_path = tmp_path / "video.mp4"
        segments = [TranscriptSegment(**seg) for seg in sample_segments]
        
        transcript = Transcript(
            video_path=video_path,
            language="en",
            segments=segments,
            full_text="Full text here"
        )
        
        assert transcript.video_path == video_path
        assert transcript.language == "en"
        assert len(transcript.segments) == 3
        assert transcript.full_text == "Full text here"
    
    def test_transcript_to_dict(self, sample_transcript):
        """Test converting transcript to dictionary"""
        result = sample_transcript.to_dict()
        
        assert "video_path" in result
        assert "language" in result
        assert "segments" in result
        assert "full_text" in result
        assert result["language"] == "en"
        assert len(result["segments"]) == 3
    
    def test_save_transcript(self, sample_transcript, tmp_path):
        """Test saving transcript to JSON file"""
        output_path = tmp_path / "transcript.json"
        sample_transcript.save(output_path)
        
        assert output_path.exists()
        
        # Verify content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data["language"] == "en"
        assert len(data["segments"]) == 3
        assert "full_text" in data
    
    def test_load_transcript(self, sample_transcript, tmp_path):
        """Test loading transcript from JSON file"""
        output_path = tmp_path / "transcript.json"
        sample_transcript.save(output_path)
        
        loaded = Transcript.load(output_path)
        
        assert loaded.language == sample_transcript.language
        assert len(loaded.segments) == len(sample_transcript.segments)
        assert loaded.full_text == sample_transcript.full_text
    
    def test_save_load_roundtrip(self, sample_transcript, tmp_path):
        """Test save and load roundtrip preserves data"""
        output_path = tmp_path / "transcript.json"
        
        # Save
        sample_transcript.save(output_path)
        
        # Load
        loaded = Transcript.load(output_path)
        
        # Compare
        assert loaded.language == sample_transcript.language
        assert loaded.full_text == sample_transcript.full_text
        assert len(loaded.segments) == len(sample_transcript.segments)
        
        for orig, loaded_seg in zip(sample_transcript.segments, loaded.segments):
            assert orig.start == loaded_seg.start
            assert orig.end == loaded_seg.end
            assert orig.text == loaded_seg.text


# Tests for TranscriptExtractor

class TestTranscriptExtractor:
    """Tests for TranscriptExtractor class"""
    
    def test_init_default(self):
        """Test initializing extractor with defaults"""
        extractor = TranscriptExtractor()
        
        assert extractor.model_size == "base"
        assert extractor.device is None
        assert extractor.language is None
        assert extractor._model is None
    
    def test_init_custom(self):
        """Test initializing extractor with custom parameters"""
        extractor = TranscriptExtractor(
            model_size="small",
            device="cuda",
            language="es"
        )
        
        assert extractor.model_size == "small"
        assert extractor.device == "cuda"
        assert extractor.language == "es"
    
    @patch('whisper.load_model')
    def test_load_model(self, mock_load_model):
        """Test lazy loading of Whisper model"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        extractor = TranscriptExtractor(model_size="tiny")
        assert extractor._model is None
        
        extractor._load_model()
        
        assert extractor._model is not None
        mock_load_model.assert_called_once_with("tiny", device=None)
    
    @patch('whisper.load_model')
    def test_load_model_with_device(self, mock_load_model):
        """Test loading model with specific device"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        extractor = TranscriptExtractor(device="cuda")
        extractor._load_model()
        
        mock_load_model.assert_called_once_with("base", device="cuda")
    
    @patch('builtins.__import__', side_effect=ImportError("No module named 'whisper'"))
    def test_load_model_import_error(self, mock_import):
        """Test handling of missing whisper package"""
        extractor = TranscriptExtractor()
        
        with pytest.raises(ImportError, match="openai-whisper is not installed"):
            extractor._load_model()
    
    @patch('whisper.load_model')
    def test_extract_success(self, mock_load_model, tmp_path, mock_whisper_model, sample_whisper_result):
        """Test successful transcript extraction"""
        # Create a dummy video file
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        mock_load_model.return_value = mock_whisper_model
        
        extractor = TranscriptExtractor()
        transcript = extractor.extract(video_path)
        
        assert isinstance(transcript, Transcript)
        assert transcript.language == "en"
        assert len(transcript.segments) == 3
        assert transcript.full_text.startswith("Welcome to this tutorial")
    
    def test_extract_file_not_found(self):
        """Test extraction with non-existent file"""
        extractor = TranscriptExtractor()
        
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            extractor.extract("nonexistent.mp4")
    
    @patch('whisper.load_model')
    def test_extract_with_output_path(self, mock_load_model, tmp_path, mock_whisper_model):
        """Test extraction with saving to output path"""
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_path = tmp_path / "output.json"
        
        mock_load_model.return_value = mock_whisper_model
        
        extractor = TranscriptExtractor()
        transcript = extractor.extract(video_path, output_path)
        
        assert output_path.exists()
        
        # Verify saved content
        loaded = Transcript.load(output_path)
        assert loaded.language == transcript.language
    
    @patch('whisper.load_model')
    def test_extract_with_language(self, mock_load_model, tmp_path, mock_whisper_model):
        """Test extraction with specified language"""
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        mock_load_model.return_value = mock_whisper_model
        
        extractor = TranscriptExtractor(language="es")
        transcript = extractor.extract(video_path)
        
        # Verify language was passed to transcribe
        mock_whisper_model.transcribe.assert_called_once()
        call_kwargs = mock_whisper_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "es"
    
    @patch('whisper.load_model')
    def test_extract_batch(self, mock_load_model, tmp_path, mock_whisper_model):
        """Test batch extraction of multiple videos"""
        # Create dummy video files
        video_paths = [
            tmp_path / "video1.mp4",
            tmp_path / "video2.mp4",
            tmp_path / "video3.mp4",
        ]
        for path in video_paths:
            path.touch()
        
        mock_load_model.return_value = mock_whisper_model
        
        extractor = TranscriptExtractor()
        transcripts = extractor.extract_batch(video_paths)
        
        assert len(transcripts) == 3
        assert all(isinstance(t, Transcript) for t in transcripts)
    
    @patch('whisper.load_model')
    def test_extract_batch_with_output_dir(self, mock_load_model, tmp_path, mock_whisper_model):
        """Test batch extraction with output directory"""
        # Create dummy video files
        video_paths = [tmp_path / "video1.mp4", tmp_path / "video2.mp4"]
        for path in video_paths:
            path.touch()
        
        output_dir = tmp_path / "transcripts"
        
        mock_load_model.return_value = mock_whisper_model
        
        extractor = TranscriptExtractor()
        transcripts = extractor.extract_batch(video_paths, output_dir)
        
        # Verify output files were created
        assert output_dir.exists()
        assert (output_dir / "video1_transcript.json").exists()
        assert (output_dir / "video2_transcript.json").exists()
    
    @patch('whisper.load_model')
    def test_model_reuse(self, mock_load_model, tmp_path, mock_whisper_model):
        """Test that model is loaded once and reused"""
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        mock_load_model.return_value = mock_whisper_model
        
        extractor = TranscriptExtractor()
        
        # Extract twice
        extractor.extract(video_path)
        extractor.extract(video_path)
        
        # Model should only be loaded once
        mock_load_model.assert_called_once()


# Integration-style tests

class TestTranscriptExtractorIntegration:
    """Integration tests for the full workflow"""
    
    @patch('whisper.load_model')
    def test_full_workflow(self, mock_load_model, tmp_path, mock_whisper_model):
        """Test complete workflow: extract, save, load"""
        video_path = tmp_path / "tutorial.mp4"
        video_path.touch()
        output_path = tmp_path / "transcript.json"
        
        mock_load_model.return_value = mock_whisper_model
        
        # Extract
        extractor = TranscriptExtractor()
        transcript = extractor.extract(video_path, output_path)
        
        # Verify extraction
        assert len(transcript.segments) > 0
        assert transcript.full_text
        
        # Verify save
        assert output_path.exists()
        
        # Load and verify
        loaded = Transcript.load(output_path)
        assert loaded.language == transcript.language
        assert len(loaded.segments) == len(transcript.segments)

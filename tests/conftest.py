"""Shared pytest fixtures and configuration for FrameWise tests.

This module provides common fixtures used across multiple test files,
including sample data, mock objects, and test utilities.
"""

from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from framewise.core.transcript_extractor import Transcript, TranscriptSegment
from framewise.core.frame_extractor import ExtractedFrame


# Shared fixtures

@pytest.fixture
def tmp_video_file(tmp_path):
    """Create a temporary dummy video file for testing.
    
    Returns:
        Path to a temporary .mp4 file (empty, just for path testing)
    """
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    return video_path


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image.
    
    Returns:
        Path to a temporary RGB image file (640x480, red color)
    """
    image_path = tmp_path / "test_image.jpg"
    img = Image.new('RGB', (640, 480), color='red')
    img.save(image_path)
    return image_path


@pytest.fixture
def sample_transcript_segments():
    """Sample transcript segments for testing.
    
    Returns:
        List of TranscriptSegment objects with varied content
    """
    return [
        TranscriptSegment(0.0, 2.0, "Welcome to this tutorial"),
        TranscriptSegment(2.0, 4.0, "Click the export button"),
        TranscriptSegment(4.0, 6.0, "Select your file format"),
        TranscriptSegment(6.0, 8.0, "Press the save icon"),
        TranscriptSegment(8.0, 10.0, "Your file is now exported"),
    ]


@pytest.fixture
def sample_transcript_obj(tmp_video_file, sample_transcript_segments):
    """Complete Transcript object for testing.
    
    Returns:
        Transcript object with sample segments
    """
    return Transcript(
        video_path=tmp_video_file,
        language="en",
        segments=sample_transcript_segments,
        full_text=" ".join(seg.text for seg in sample_transcript_segments)
    )


@pytest.fixture
def sample_extracted_frames(tmp_path, sample_transcript_segments):
    """Sample extracted frames for testing.
    
    Returns:
        List of ExtractedFrame objects with images and transcripts
    """
    frames = []
    for i, segment in enumerate(sample_transcript_segments[:3]):
        # Create frame image
        frame_path = tmp_path / f"frame_{i:04d}.jpg"
        img = Image.new('RGB', (640, 480), color='blue')
        img.save(frame_path)
        
        frame = ExtractedFrame(
            frame_id=f"frame_{i:04d}",
            path=frame_path,
            timestamp=segment.start + 1.0,
            transcript_segment=segment,
            extraction_reason="keyword:click" if i % 2 == 0 else "scene_change",
            scene_change_score=0.5 + (i * 0.1),
            quality_score=0.8 + (i * 0.05)
        )
        frames.append(frame)
    
    return frames


@pytest.fixture
def sample_frame_embeddings():
    """Sample frame embeddings for vector store testing.
    
    Returns:
        List of embedding dictionaries
    """
    return [
        {
            "frame_id": f"frame_{i:04d}",
            "timestamp": float(i * 2),
            "text": f"Sample text {i}",
            "frame_path": f"frames/frame_{i:04d}.jpg",
            "extraction_reason": "test",
            "quality_score": 0.9,
            "image_embedding": np.random.rand(512),
            "text_embedding": np.random.rand(384),
        }
        for i in range(5)
    ]


# Test utilities

def assert_embeddings_valid(embeddings, expected_count=None):
    """Helper to validate embedding structure.
    
    Args:
        embeddings: List of embedding dictionaries
        expected_count: Expected number of embeddings (optional)
    """
    if expected_count is not None:
        assert len(embeddings) == expected_count
    
    for emb in embeddings:
        assert "frame_id" in emb
        assert "timestamp" in emb
        assert "image_embedding" in emb
        assert "text_embedding" in emb
        assert isinstance(emb["image_embedding"], np.ndarray)
        assert isinstance(emb["text_embedding"], np.ndarray)


def assert_transcript_valid(transcript):
    """Helper to validate Transcript structure.
    
    Args:
        transcript: Transcript object to validate
    """
    assert isinstance(transcript, Transcript)
    assert isinstance(transcript.video_path, Path)
    assert isinstance(transcript.language, str)
    assert isinstance(transcript.segments, list)
    assert isinstance(transcript.full_text, str)
    assert len(transcript.segments) > 0


def assert_frames_valid(frames, expected_count=None):
    """Helper to validate ExtractedFrame list.
    
    Args:
        frames: List of ExtractedFrame objects
        expected_count: Expected number of frames (optional)
    """
    if expected_count is not None:
        assert len(frames) == expected_count
    
    for frame in frames:
        assert isinstance(frame, ExtractedFrame)
        assert isinstance(frame.frame_id, str)
        assert isinstance(frame.path, Path)
        assert isinstance(frame.timestamp, float)
        assert 0.0 <= frame.quality_score <= 1.0
        assert 0.0 <= frame.scene_change_score <= 1.0

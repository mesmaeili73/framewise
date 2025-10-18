"""Frame extraction from video files with intelligent keyframe selection.

This module provides functionality to extract keyframes from video files using
multiple strategies: scene detection, transcript alignment, or a hybrid approach.
It intelligently selects the most important visual moments from tutorial videos.

Example:
    Basic usage::

        from framewise import FrameExtractor, TranscriptExtractor
        
        # Extract transcript first
        transcript = TranscriptExtractor().extract("video.mp4")
        
        # Extract frames using hybrid strategy
        extractor = FrameExtractor(strategy="hybrid")
        frames = extractor.extract("video.mp4", transcript=transcript)
        
        for frame in frames:
            print(f"{frame.timestamp}s: {frame.extraction_reason}")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass
import json
import cv2
import numpy as np
from PIL import Image
from loguru import logger

from framewise.core.transcript_extractor import Transcript, TranscriptSegment


@dataclass
class ExtractedFrame:
    """A single extracted frame with metadata.
    
    Represents a keyframe extracted from a video, including its location,
    timing, associated transcript segment, and quality metrics.
    
    Attributes:
        frame_id: Unique identifier for this frame (e.g., "frame_0001").
        path: Path to the saved frame image file.
        timestamp: Time in seconds when this frame appears in the video.
        transcript_segment: Associated transcript segment, if available.
        extraction_reason: Why this frame was extracted (e.g., "scene_change",
            "keyword:click").
        scene_change_score: Score indicating magnitude of scene change (0-1).
        quality_score: Quality assessment score (0-1, higher is better).
    
    Example:
        >>> frame = ExtractedFrame(
        ...     frame_id="frame_0001",
        ...     path=Path("frames/frame_0001.jpg"),
        ...     timestamp=12.5,
        ...     extraction_reason="keyword:click",
        ...     quality_score=0.85
        ... )
        >>> print(f"Frame at {frame.timestamp}s: {frame.extraction_reason}")
        Frame at 12.5s: keyword:click
    """
    
    frame_id: str
    path: Path
    timestamp: float
    transcript_segment: Optional[TranscriptSegment] = None
    extraction_reason: str = "unknown"
    scene_change_score: float = 0.0
    quality_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Union[str, float, Dict, None]]:
        """Convert frame to dictionary format.
        
        Returns:
            Dictionary containing all frame metadata.
            
        Example:
            >>> frame.to_dict()
            {
                'frame_id': 'frame_0001',
                'path': 'frames/frame_0001.jpg',
                'timestamp': 12.5,
                'extraction_reason': 'keyword:click',
                'quality_score': 0.85
            }
        """
        return {
            "frame_id": self.frame_id,
            "path": str(self.path),
            "timestamp": self.timestamp,
            "transcript_segment": self.transcript_segment.to_dict() if self.transcript_segment else None,
            "extraction_reason": self.extraction_reason,
            "scene_change_score": self.scene_change_score,
            "quality_score": self.quality_score,
        }


class FrameExtractor:
    """Extract keyframes from videos using intelligent strategies.
    
    This class provides multiple strategies for extracting important frames
    from tutorial videos:
    
    - **Scene Detection**: Identifies visual transitions and changes
    - **Transcript Alignment**: Extracts frames when action keywords are mentioned
    - **Hybrid**: Combines both approaches for optimal coverage
    
    The extractor also performs quality assessment to filter out blurry or
    low-quality frames.
    
    Attributes:
        strategy: The extraction strategy being used.
        max_frames_per_video: Maximum number of frames to extract per video.
        scene_threshold: Threshold for scene change detection (0-1).
        quality_threshold: Minimum quality score for frames (0-1).
        ACTION_KEYWORDS: List of keywords indicating important moments.
    
    Example:
        Scene detection only::
        
            extractor = FrameExtractor(strategy="scene", scene_threshold=0.3)
            frames = extractor.extract("video.mp4")
        
        Transcript-based extraction::
        
            transcript = TranscriptExtractor().extract("video.mp4")
            extractor = FrameExtractor(strategy="transcript")
            frames = extractor.extract("video.mp4", transcript=transcript)
        
        Hybrid approach (recommended)::
        
            extractor = FrameExtractor(
                strategy="hybrid",
                max_frames_per_video=20,
                quality_threshold=0.5
            )
            frames = extractor.extract("video.mp4", transcript=transcript)
    """
    
    # Action keywords that indicate important moments in tutorials
    ACTION_KEYWORDS = [
        "click", "select", "choose", "open", "close", "press", "tap",
        "drag", "drop", "scroll", "type", "enter", "delete", "save",
        "export", "import", "upload", "download", "copy", "paste",
        "button", "menu", "icon", "tab", "window", "dialog", "popup"
    ]
    
    def __init__(
        self,
        strategy: str = "hybrid",
        max_frames_per_video: int = 20,
        scene_threshold: float = 0.3,
        quality_threshold: float = 0.5,
    ) -> None:
        """Initialize the frame extractor.
        
        Args:
            strategy: Extraction strategy to use. Options:
                - 'scene': Extract frames at scene changes only
                - 'transcript': Extract frames when action keywords are mentioned
                - 'hybrid': Combine both strategies (recommended)
                Defaults to 'hybrid'.
            max_frames_per_video: Maximum number of frames to extract from each
                video. If more candidates are found, they will be evenly sampled.
                Defaults to 20.
            scene_threshold: Threshold for scene change detection (0-1).
                Higher values = only major scene changes. Lower values = more
                sensitive to changes. Defaults to 0.3.
            quality_threshold: Minimum quality score for frames (0-1).
                Frames below this threshold will be filtered out. Higher values
                = stricter quality requirements. Defaults to 0.5.
        
        Raises:
            ValueError: If strategy is not one of 'scene', 'transcript', or 'hybrid'.
        
        Example:
            >>> # Strict quality, fewer frames
            >>> extractor = FrameExtractor(
            ...     strategy="hybrid",
            ...     max_frames_per_video=10,
            ...     quality_threshold=0.7
            ... )
            
            >>> # More sensitive to scene changes
            >>> extractor = FrameExtractor(
            ...     strategy="scene",
            ...     scene_threshold=0.2
            ... )
        """
        if strategy not in ["scene", "transcript", "hybrid"]:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                "Must be 'scene', 'transcript', or 'hybrid'"
            )
        
        self.strategy = strategy
        self.max_frames_per_video = max_frames_per_video
        self.scene_threshold = scene_threshold
        self.quality_threshold = quality_threshold
    
    def extract(
        self,
        video_path: Union[str, Path],
        transcript: Optional[Transcript] = None,
        output_dir: Union[str, Path] = "frames",
    ) -> List[ExtractedFrame]:
        """Extract keyframes from a video file.
        
        Processes the video using the configured strategy to identify and extract
        important frames. Frames are saved as JPEG images with metadata.
        
        Args:
            video_path: Path to the video file to process. Supports common formats
                (mp4, avi, mov, mkv, etc.).
            transcript: Optional Transcript object for transcript-based or hybrid
                extraction. Required if strategy is 'transcript' or 'hybrid'.
                Defaults to None.
            output_dir: Directory where extracted frames and metadata will be saved.
                Will be created if it doesn't exist. Defaults to "frames".
        
        Returns:
            List of ExtractedFrame objects, sorted by timestamp. Each frame includes
            the image path, timestamp, quality score, and extraction reason.
        
        Raises:
            FileNotFoundError: If the video file doesn't exist.
            ValueError: If strategy is 'transcript' but no transcript is provided,
                or if the video file cannot be opened.
            RuntimeError: If frame extraction fails.
        
        Example:
            >>> extractor = FrameExtractor(strategy="hybrid")
            >>> transcript = TranscriptExtractor().extract("tutorial.mp4")
            >>> frames = extractor.extract(
            ...     "tutorial.mp4",
            ...     transcript=transcript,
            ...     output_dir="output/frames"
            ... )
            >>> print(f"Extracted {len(frames)} frames")
            Extracted 15 frames
            >>> for frame in frames[:3]:
            ...     print(f"{frame.timestamp}s: {frame.extraction_reason}")
            2.5s: scene_change
            8.3s: keyword:click
            15.7s: scene_change
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Extracting frames from: {video_path.name}")
        logger.info(f"Strategy: {self.strategy}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {duration:.1f}s, {fps:.1f} fps, {total_frames} frames")
        
        # Extract frames based on strategy
        if self.strategy == "scene":
            candidate_timestamps = self._extract_by_scene_change(cap, fps)
        elif self.strategy == "transcript":
            if transcript is None:
                raise ValueError("Transcript required for 'transcript' strategy")
            candidate_timestamps = self._extract_by_transcript(transcript)
        elif self.strategy == "hybrid":
            scene_timestamps = self._extract_by_scene_change(cap, fps)
            transcript_timestamps = self._extract_by_transcript(transcript) if transcript else []
            # Combine and deduplicate
            candidate_timestamps = self._merge_timestamps(scene_timestamps, transcript_timestamps)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Limit number of frames
        if len(candidate_timestamps) > self.max_frames_per_video:
            # Keep evenly distributed frames
            step = len(candidate_timestamps) / self.max_frames_per_video
            candidate_timestamps = [
                candidate_timestamps[int(i * step)] 
                for i in range(self.max_frames_per_video)
            ]
        
        logger.info(f"Extracting {len(candidate_timestamps)} frames")
        
        # Extract and save frames
        extracted_frames = []
        for idx, timestamp_info in enumerate(candidate_timestamps):
            if isinstance(timestamp_info, tuple):
                timestamp, reason, score = timestamp_info
            else:
                timestamp, reason, score = timestamp_info, "unknown", 0.0
            
            frame = self._extract_frame_at_timestamp(cap, timestamp, fps)
            if frame is None:
                continue
            
            # Check quality
            quality = self._assess_frame_quality(frame)
            if quality < self.quality_threshold:
                logger.debug(f"Skipping low quality frame at {timestamp:.1f}s")
                continue
            
            # Save frame
            frame_id = f"frame_{idx:04d}"
            frame_filename = f"{frame_id}_t{timestamp:07.1f}s.jpg"
            frame_path = output_dir / frame_filename
            
            cv2.imwrite(str(frame_path), frame)
            
            # Find associated transcript segment
            segment = self._find_transcript_segment(transcript, timestamp) if transcript else None
            
            extracted_frame = ExtractedFrame(
                frame_id=frame_id,
                path=frame_path,
                timestamp=timestamp,
                transcript_segment=segment,
                extraction_reason=reason,
                scene_change_score=score,
                quality_score=quality,
            )
            
            extracted_frames.append(extracted_frame)
            logger.debug(f"Extracted: {frame_filename}")
        
        cap.release()
        
        # Save metadata
        self._save_metadata(extracted_frames, output_dir)
        
        logger.success(f"Extracted {len(extracted_frames)} frames to {output_dir}")
        return extracted_frames
    
    def _extract_by_scene_change(
        self,
        cap: cv2.VideoCapture,
        fps: float
    ) -> List[Tuple[float, str, float]]:
        """Extract frames at scene changes using visual difference detection.
        
        Analyzes consecutive frames to detect significant visual changes that
        indicate scene transitions, UI changes, or important visual moments.
        
        Args:
            cap: OpenCV VideoCapture object for the video.
            fps: Frames per second of the video.
        
        Returns:
            List of tuples containing (timestamp, reason, score) for each
            detected scene change.
        
        Note:
            Samples every 30 frames for efficiency. Adjust sampling rate for
            different video types or frame rates.
        """
        timestamps = []
        prev_frame = None
        frame_idx = 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if prev_frame is not None:
                # Calculate scene change score
                score = self._calculate_scene_change(prev_frame, frame)
                
                if score > self.scene_threshold:
                    timestamp = frame_idx / fps
                    timestamps.append((timestamp, "scene_change", score))
            
            prev_frame = frame
            frame_idx += 1
            
            # Sample every N frames for efficiency
            if frame_idx % 30 == 0:  # Check every 30 frames (~0.5s at 60fps)
                continue
        
        return timestamps
    
    def _extract_by_transcript(
        self,
        transcript: Transcript
    ) -> List[Tuple[float, str, float]]:
        """Extract frames at important transcript moments.
        
        Identifies segments containing action keywords (e.g., "click", "select")
        and extracts frames at those moments. This captures frames when the
        narrator is describing actions.
        
        Args:
            transcript: Transcript object with timestamped segments.
        
        Returns:
            List of tuples containing (timestamp, reason, score) for each
            keyword match. Timestamp is the midpoint of the segment.
        
        Note:
            Only extracts one frame per segment, even if multiple keywords
            are present.
        """
        timestamps = []
        
        for segment in transcript.segments:
            text_lower = segment.text.lower()
            
            # Check for action keywords
            for keyword in self.ACTION_KEYWORDS:
                if keyword in text_lower:
                    # Extract at the middle of the segment
                    timestamp = (segment.start + segment.end) / 2
                    timestamps.append((timestamp, f"keyword:{keyword}", 1.0))
                    break  # One frame per segment
        
        return timestamps
    
    def _merge_timestamps(
        self,
        scene_timestamps: List[Tuple[float, str, float]],
        transcript_timestamps: List[Tuple[float, str, float]],
        merge_window: float = 2.0
    ) -> List[Tuple[float, str, float]]:
        """Merge timestamps from different strategies, removing duplicates.
        
        Combines timestamps from scene detection and transcript analysis,
        removing duplicates that are too close together. When timestamps
        are within the merge window, keeps the one with the higher score.
        
        Args:
            scene_timestamps: Timestamps from scene change detection.
            transcript_timestamps: Timestamps from transcript analysis.
            merge_window: Time window in seconds for considering timestamps
                as duplicates. Defaults to 2.0 seconds.
        
        Returns:
            Merged and deduplicated list of timestamps, sorted by time.
        
        Example:
            >>> scene_ts = [(1.0, "scene_change", 0.8), (5.0, "scene_change", 0.6)]
            >>> trans_ts = [(1.5, "keyword:click", 1.0), (10.0, "keyword:save", 1.0)]
            >>> merged = extractor._merge_timestamps(scene_ts, trans_ts)
            >>> # (1.5, "keyword:click", 1.0) kept over (1.0, ...) due to higher score
        """
        all_timestamps = scene_timestamps + transcript_timestamps
        all_timestamps.sort(key=lambda x: x[0])
        
        merged = []
        for timestamp, reason, score in all_timestamps:
            # Check if too close to previous timestamp
            if merged and abs(merged[-1][0] - timestamp) < merge_window:
                # Keep the one with higher score
                if score > merged[-1][2]:
                    merged[-1] = (timestamp, reason, score)
            else:
                merged.append((timestamp, reason, score))
        
        return merged
    
    def _extract_frame_at_timestamp(
        self,
        cap: cv2.VideoCapture,
        timestamp: float,
        fps: float
    ) -> Optional[np.ndarray]:
        """Extract a single frame at the given timestamp.
        
        Args:
            cap: OpenCV VideoCapture object.
            timestamp: Time in seconds to extract the frame.
            fps: Frames per second of the video.
        
        Returns:
            Frame as numpy array (BGR format), or None if extraction fails.
        """
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if not ret:
            return None
        
        return frame
    
    def _calculate_scene_change(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
        """Calculate scene change score between two frames.
        
        Computes the visual difference between consecutive frames using
        grayscale conversion and absolute difference. Higher scores indicate
        more significant visual changes.
        
        Args:
            frame1: First frame as numpy array (BGR format).
            frame2: Second frame as numpy array (BGR format).
        
        Returns:
            Scene change score between 0 and 1, where:
            - 0 = identical frames
            - 1 = completely different frames
            - Typical scene changes: 0.3-0.7
        
        Note:
            Frames are resized to 320x240 for faster computation without
            significant loss in change detection accuracy.
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Resize for faster computation
        gray1 = cv2.resize(gray1, (320, 240))
        gray2 = cv2.resize(gray2, (320, 240))
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Calculate mean difference (normalized to 0-1)
        score = np.mean(diff) / 255.0
        
        return score
    
    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        """Assess frame quality using blur detection.
        
        Uses Laplacian variance to detect image sharpness. Blurry frames
        (e.g., during camera movement) receive lower scores.
        
        Args:
            frame: Frame as numpy array (BGR format).
        
        Returns:
            Quality score between 0 and 1, where:
            - 0 = very blurry
            - 1 = very sharp
            - Typical sharp frames: 0.5-1.0
            - Typical blurry frames: 0.0-0.3
        
        Note:
            Uses Laplacian variance with normalization. Typical values:
            - <100 = blurry
            - 100-500 = acceptable
            - >500 = sharp
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (higher = sharper)
        # Typical values: <100 = blurry, >500 = sharp
        quality = min(laplacian_var / 500.0, 1.0)
        
        return quality
    
    def _find_transcript_segment(
        self,
        transcript: Transcript,
        timestamp: float
    ) -> Optional[TranscriptSegment]:
        """Find the transcript segment that contains or is closest to the timestamp.
        
        Args:
            transcript: Transcript object with segments.
            timestamp: Time in seconds to find the segment for.
        
        Returns:
            TranscriptSegment if found within 2 seconds, None otherwise.
        
        Note:
            First tries to find a segment that contains the timestamp.
            If none found, returns the closest segment within 2 seconds.
        """
        # Try to find segment containing the timestamp
        for segment in transcript.segments:
            if segment.start <= timestamp <= segment.end:
                return segment
        
        # Find closest segment if not within any
        closest = min(
            transcript.segments,
            key=lambda s: min(abs(s.start - timestamp), abs(s.end - timestamp))
        )
        
        # Only return if reasonably close (within 2 seconds)
        if min(abs(closest.start - timestamp), abs(closest.end - timestamp)) < 2.0:
            return closest
        
        return None
    
    def _save_metadata(
        self,
        frames: List[ExtractedFrame],
        output_dir: Path
    ) -> None:
        """Save frame metadata to JSON file.
        
        Creates a metadata.json file in the output directory containing
        information about all extracted frames.
        
        Args:
            frames: List of extracted frames.
            output_dir: Directory where metadata.json will be saved.
        
        Raises:
            IOError: If the metadata file cannot be written.
        """
        metadata = {
            "total_frames": len(frames),
            "frames": [frame.to_dict() for frame in frames]
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved metadata to {metadata_path}")

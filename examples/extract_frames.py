"""
Example: Extract frames from a video with transcript alignment

This example shows how to use the FrameExtractor to extract keyframes
from tutorial videos, intelligently aligned with transcript timestamps.
"""

from pathlib import Path
from loguru import logger
from framewise import TranscriptExtractor, FrameExtractor, Transcript


def main():
    """
    Extract frames from a video using different strategies
    """
    
    # ========================================
    # CONFIGURE THIS: Put your video path here
    # ========================================
    VIDEO_PATH = "path/to/your/tutorial.mp4"
    
    # Check if file exists
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        logger.error(f"❌ Video file not found: {VIDEO_PATH}")
        logger.info("\n📝 To test frame extraction:")
        logger.info("1. Update VIDEO_PATH with your video file")
        logger.info("2. Run: python examples/extract_frames.py")
        return
    
    logger.info("🎬 FrameWise Frame Extraction Demo")
    logger.info("=" * 50)
    
    # Example 1: Extract frames with transcript (hybrid strategy)
    logger.info("\n📖 Example 1: Hybrid Strategy (Scene + Transcript)")
    logger.info("-" * 50)
    
    # First, extract transcript
    logger.info("Step 1: Extracting transcript...")
    transcript_extractor = TranscriptExtractor(model_size="base")
    transcript = transcript_extractor.extract(VIDEO_PATH)
    logger.success(f"✓ Transcript extracted: {len(transcript.segments)} segments")
    
    # Then, extract frames using transcript
    logger.info("\nStep 2: Extracting frames with hybrid strategy...")
    frame_extractor = FrameExtractor(
        strategy="hybrid",
        max_frames_per_video=15,
        scene_threshold=0.3,
        quality_threshold=0.5
    )
    
    frames = frame_extractor.extract(
        video_path=VIDEO_PATH,
        transcript=transcript,
        output_dir="frames_hybrid"
    )
    
    logger.success(f"✓ Extracted {len(frames)} frames")
    
    # Show results
    logger.info("\n📊 Extracted Frames:")
    for i, frame in enumerate(frames[:5]):
        logger.info(f"\n  Frame {i+1}:")
        logger.info(f"    Path: {frame.path.name}")
        logger.info(f"    Time: {frame.timestamp:.1f}s")
        logger.info(f"    Reason: {frame.extraction_reason}")
        logger.info(f"    Quality: {frame.quality_score:.2f}")
        if frame.transcript_segment:
            logger.info(f"    Text: '{frame.transcript_segment.text}'")
    
    if len(frames) > 5:
        logger.info(f"\n  ... and {len(frames) - 5} more frames")
    
    # Example 2: Scene-only strategy
    logger.info("\n" + "=" * 50)
    logger.info("📖 Example 2: Scene Change Strategy")
    logger.info("-" * 50)
    
    frame_extractor_scene = FrameExtractor(
        strategy="scene",
        max_frames_per_video=10,
        scene_threshold=0.4  # Higher threshold = fewer frames
    )
    
    frames_scene = frame_extractor_scene.extract(
        video_path=VIDEO_PATH,
        output_dir="frames_scene"
    )
    
    logger.success(f"✓ Extracted {len(frames_scene)} frames (scene changes only)")
    
    # Example 3: Transcript-only strategy
    logger.info("\n" + "=" * 50)
    logger.info("📖 Example 3: Transcript-Aligned Strategy")
    logger.info("-" * 50)
    
    frame_extractor_transcript = FrameExtractor(
        strategy="transcript",
        max_frames_per_video=20
    )
    
    frames_transcript = frame_extractor_transcript.extract(
        video_path=VIDEO_PATH,
        transcript=transcript,
        output_dir="frames_transcript"
    )
    
    logger.success(f"✓ Extracted {len(frames_transcript)} frames (action keywords only)")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Summary")
    logger.info("=" * 50)
    logger.info(f"  Hybrid strategy:     {len(frames)} frames")
    logger.info(f"  Scene-only:          {len(frames_scene)} frames")
    logger.info(f"  Transcript-only:     {len(frames_transcript)} frames")
    
    logger.info("\n💾 Output directories:")
    logger.info("  - frames_hybrid/     (recommended)")
    logger.info("  - frames_scene/")
    logger.info("  - frames_transcript/")
    
    logger.info("\n🎯 Each directory contains:")
    logger.info("  - Extracted frame images (.jpg)")
    logger.info("  - metadata.json (frame info + transcript links)")
    
    logger.info("\n📚 Next steps:")
    logger.info("  1. Check the frames/ directories")
    logger.info("  2. Review metadata.json for frame details")
    logger.info("  3. Try different strategies and thresholds")
    logger.info("  4. Ready for Phase 3: Embeddings!")


def show_usage_examples():
    """Show code examples"""
    
    logger.info("\n📖 Usage Examples:")
    logger.info("=" * 50)
    
    logger.info("\n1️⃣  Basic Frame Extraction:")
    logger.info("""
    from framewise import FrameExtractor
    
    extractor = FrameExtractor(strategy="scene")
    frames = extractor.extract("video.mp4", output_dir="frames/")
    
    print(f"Extracted {len(frames)} frames")
    """)
    
    logger.info("\n2️⃣  With Transcript (Recommended):")
    logger.info("""
    from framewise import TranscriptExtractor, FrameExtractor
    
    # Get transcript first
    transcript_ext = TranscriptExtractor()
    transcript = transcript_ext.extract("video.mp4")
    
    # Extract frames with transcript alignment
    frame_ext = FrameExtractor(strategy="hybrid")
    frames = frame_ext.extract(
        "video.mp4",
        transcript=transcript,
        output_dir="frames/"
    )
    """)
    
    logger.info("\n3️⃣  Access Frame Data:")
    logger.info("""
    for frame in frames:
        print(f"Time: {frame.timestamp}s")
        print(f"Path: {frame.path}")
        print(f"Reason: {frame.extraction_reason}")
        if frame.transcript_segment:
            print(f"Text: {frame.transcript_segment.text}")
    """)
    
    logger.info("\n4️⃣  Custom Settings:")
    logger.info("""
    extractor = FrameExtractor(
        strategy="hybrid",
        max_frames_per_video=20,      # Limit frames
        scene_threshold=0.3,           # Scene sensitivity
        quality_threshold=0.5          # Min quality
    )
    """)


if __name__ == "__main__":
    logger.info("🎬 FrameWise - Frame Extraction Demo")
    logger.info("=" * 50)
    
    # Show usage examples
    show_usage_examples()
    
    logger.info("\n" + "=" * 50)
    logger.info("🧪 Running Demo...")
    logger.info("=" * 50)
    
    # Run the demo
    main()

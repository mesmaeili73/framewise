"""
Simple script to test the TranscriptExtractor

This script will help you understand what the transcript extractor does
by running it on a sample video.
"""

from pathlib import Path
from loguru import logger
<<<<<<< HEAD
<<<<<<< HEAD
from framewise import TranscriptExtractor, FrameExtractor
=======
from framewise import TranscriptExtractor
>>>>>>> 4b83543 (fix: Workaround for triton dependency on macOS)
=======
from framewise import TranscriptExtractor, FrameExtractor
>>>>>>> e380357 (feat: Add frame extraction to test_it_yourself.py)

def test_with_sample_video():
    """
    Test transcript extraction with a video file
    
    INSTRUCTIONS:
    1. Place a video file in this directory (any format: mp4, avi, mov, etc.)
    2. Update VIDEO_PATH below with your video filename
    3. Run: python test_it_yourself.py
    """
    
    # ========================================
    # CONFIGURE THIS: Put your video filename here
    # ========================================
<<<<<<< HEAD
<<<<<<< HEAD
    VIDEO_PATH = "/Users/mes/Desktop/Screen Recording 2025-10-17 at 15.45.45.mov"  # Change this to your video file
=======
    VIDEO_PATH = "your_video.mp4"  # Change this to your video file
>>>>>>> 4b83543 (fix: Workaround for triton dependency on macOS)
=======
    VIDEO_PATH = "/Users/mes/Desktop/Screen Recording 2025-10-17 at 15.45.45.mov"  # Change this to your video file
>>>>>>> 4338149 (feat: Add intelligent frame extraction with scene detection)
    
    # Check if video exists
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        logger.error(f"❌ Video file not found: {VIDEO_PATH}")
        logger.info("\n📝 To test the transcript extractor:")
        logger.info("1. Place a video file in this directory")
        logger.info("2. Edit this file and change VIDEO_PATH to your video filename")
        logger.info("3. Run: python test_it_yourself.py")
        logger.info("\n💡 Tip: Use a short video (< 1 minute) for faster testing")
        return
    
    logger.info("🎬 FrameWise Transcript Extractor Test")
    logger.info("=" * 50)
    
    # Initialize the extractor
    logger.info("\n📦 Initializing TranscriptExtractor...")
    logger.info("   Model: base (good balance of speed and accuracy)")
    logger.info("   Device: CPU (will use GPU if available)")
    
    extractor = TranscriptExtractor(
        model_size="base",  # Options: tiny, base, small, medium, large
        language="en"       # Set to None for auto-detection
    )
    
    # Extract transcript
    logger.info(f"\n🎙️  Extracting transcript from: {VIDEO_PATH}")
    logger.info("   This may take a minute...")
    
    try:
        transcript = extractor.extract(
            video_path=VIDEO_PATH,
<<<<<<< HEAD
<<<<<<< HEAD
            output_path="test_outputs/transcript.json"  # Save to JSON
=======
            output_path="test_transcript.json"  # Save to JSON
>>>>>>> 4b83543 (fix: Workaround for triton dependency on macOS)
=======
            output_path="test_outputs/transcript.json"  # Save to JSON
>>>>>>> 858dc50 (refactor: Move test outputs to test_outputs/ directory)
        )
        
        logger.success("\n✅ Transcript extracted successfully!")
        
        # Display results
        logger.info("\n📊 Results:")
        logger.info(f"   Language detected: {transcript.language}")
        logger.info(f"   Total segments: {len(transcript.segments)}")
        logger.info(f"   Video: {transcript.video_path.name}")
        
        # Show first few segments
        logger.info("\n📝 First few segments:")
        for i, segment in enumerate(transcript.segments[:5]):
            logger.info(
                f"   [{segment.start:.1f}s - {segment.end:.1f}s]: "
                f"{segment.text}"
            )
        
        if len(transcript.segments) > 5:
            logger.info(f"   ... and {len(transcript.segments) - 5} more segments")
        
        # Show full text preview
        logger.info("\n📄 Full transcript preview:")
        preview = transcript.full_text[:300]
        if len(transcript.full_text) > 300:
            preview += "..."
        logger.info(f"   {preview}")
        
        # Saved file info
<<<<<<< HEAD
<<<<<<< HEAD
        logger.info("\n💾 Transcript saved to: test_outputs/transcript.json")
=======
        logger.info("\n💾 Transcript saved to: test_transcript.json")
>>>>>>> 4b83543 (fix: Workaround for triton dependency on macOS)
=======
        logger.info("\n💾 Transcript saved to: test_outputs/transcript.json")
>>>>>>> 858dc50 (refactor: Move test outputs to test_outputs/ directory)
        logger.info("   You can open this file to see the complete transcript")
        
        # What you can do with this
        logger.info("\n🎯 What you can do with this transcript:")
        logger.info("   1. Search for specific words/phrases")
        logger.info("   2. Extract frames at specific timestamps")
        logger.info("   3. Feed to LLM for question answering")
        logger.info("   4. Generate automatic captions/subtitles")
        
        # Example: Search for a word
        logger.info("\n🔍 Example: Searching for keywords...")
        search_word = "the"  # Change this to search for different words
        matching_segments = [
            seg for seg in transcript.segments 
            if search_word.lower() in seg.text.lower()
        ]
        logger.info(f"   Found '{search_word}' in {len(matching_segments)} segments")
        if matching_segments:
            logger.info(f"   First occurrence at {matching_segments[0].start:.1f}s:")
            logger.info(f"   '{matching_segments[0].text}'")
        
    except Exception as e:
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 4338149 (feat: Add intelligent frame extraction with scene detection)
        error_msg = str(e)
        logger.error(f"\n❌ Error during extraction: {error_msg}")
        
        # Check for common issues
        if "Failed to load audio" in error_msg or "does not contain any stream" in error_msg:
            logger.warning("\n⚠️  This video appears to have no audio track!")
            logger.info("   Screen recordings often don't include audio.")
            logger.info("   Whisper needs audio to create transcripts.")
            logger.info("\n💡 Solutions:")
            logger.info("   1. Use a video with spoken audio/narration")
            logger.info("   2. Try a tutorial video with voice-over")
            logger.info("   3. Record a new screen recording with microphone enabled")
        else:
            logger.info("\n💡 Troubleshooting:")
            logger.info("   1. Make sure the video file is valid")
            logger.info("   2. Check that you have enough disk space")
            logger.info("   3. Try with a shorter video first")
            logger.info("   4. Ensure ffmpeg is installed: brew install ffmpeg")
<<<<<<< HEAD
        return
    
    # Now test frame extraction!
    logger.info("\n" + "=" * 50)
    logger.info("🎬 Testing Frame Extraction")
    logger.info("=" * 50)
    
    try:
        logger.info("\n📦 Initializing FrameExtractor...")
        logger.info("   Strategy: hybrid (scene + transcript)")
        logger.info("   Max frames: 10")
        
        frame_extractor = FrameExtractor(
            strategy="hybrid",
            max_frames_per_video=10,
            scene_threshold=0.3,
            quality_threshold=0.1  # Lower threshold to accept more frames
<<<<<<< HEAD
        )
        
        logger.info(f"\n🖼️  Extracting frames from: {VIDEO_PATH}")
        logger.info("   This will analyze the video for key moments...")
        
        frames = frame_extractor.extract(
            video_path=VIDEO_PATH,
            transcript=transcript,
            output_dir="test_outputs/frames"
        )
        
        logger.success(f"\n✅ Extracted {len(frames)} frames!")
        
        # Show frame details
        logger.info("\n📊 Extracted Frames:")
        for i, frame in enumerate(frames[:5]):
            logger.info(f"\n  Frame {i+1}:")
            logger.info(f"    File: {frame.path.name}")
            logger.info(f"    Time: {frame.timestamp:.1f}s")
            logger.info(f"    Reason: {frame.extraction_reason}")
            logger.info(f"    Quality: {frame.quality_score:.2f}")
            if frame.transcript_segment:
                logger.info(f"    Text: '{frame.transcript_segment.text}'")
        
        if len(frames) > 5:
            logger.info(f"\n  ... and {len(frames) - 5} more frames")
        
        logger.info("\n💾 Frames saved to: test_outputs/frames/")
        logger.info("   - Frame images (.jpg)")
        logger.info("   - metadata.json (frame details)")
        
        logger.info("\n🎯 What you can do with these frames:")
        logger.info("   1. View the extracted screenshots")
        logger.info("   2. See which moments were captured")
        logger.info("   3. Check metadata.json for frame-transcript links")
        logger.info("   4. Ready for Phase 3: Embeddings!")
        
    except Exception as e:
        logger.error(f"\n❌ Error during frame extraction: {e}")
        logger.info("\n💡 This is okay - frame extraction is optional")
        logger.info("   The transcript extraction still worked!")
    
    logger.info("\n" + "=" * 50)
    logger.success("✅ Complete Pipeline Test Finished!")
    logger.info("\n📚 What we tested:")
    logger.info("   ✓ Transcript extraction (audio → text)")
    logger.info("   ✓ Frame extraction (video → keyframes)")
    logger.info("   ✓ Transcript-frame alignment")
    logger.info("\n📁 Output files:")
    logger.info("   - test_outputs/transcript.json (transcript data)")
    logger.info("   - test_outputs/frames/ (extracted frames)")
    logger.info("\n🚀 Next: Try with your own tutorial videos!")
=======
        logger.error(f"\n❌ Error during extraction: {e}")
        logger.info("\n💡 Troubleshooting:")
        logger.info("   1. Make sure the video file is valid")
        logger.info("   2. Check that you have enough disk space")
        logger.info("   3. Try with a shorter video first")
=======
>>>>>>> 4338149 (feat: Add intelligent frame extraction with scene detection)
        return
    
    # Now test frame extraction!
    logger.info("\n" + "=" * 50)
<<<<<<< HEAD
    logger.success("✅ Test complete!")
    logger.info("\n📚 Next steps:")
    logger.info("   1. Check test_transcript.json to see the full output")
    logger.info("   2. Try with different videos")
    logger.info("   3. Experiment with different model sizes")
    logger.info("   4. Ready to add frame extraction next!")
>>>>>>> 4b83543 (fix: Workaround for triton dependency on macOS)
=======
    logger.info("🎬 Testing Frame Extraction")
    logger.info("=" * 50)
    
    try:
        logger.info("\n📦 Initializing FrameExtractor...")
        logger.info("   Strategy: hybrid (scene + transcript)")
        logger.info("   Max frames: 10")
        
        frame_extractor = FrameExtractor(
            strategy="hybrid",
            max_frames_per_video=10,
            scene_threshold=0.3,
            quality_threshold=0.5
=======
>>>>>>> 85b4cf8 (feat: Lower quality threshold for testing)
        )
        
        logger.info(f"\n🖼️  Extracting frames from: {VIDEO_PATH}")
        logger.info("   This will analyze the video for key moments...")
        
        frames = frame_extractor.extract(
            video_path=VIDEO_PATH,
            transcript=transcript,
            output_dir="test_outputs/frames"
        )
        
        logger.success(f"\n✅ Extracted {len(frames)} frames!")
        
        # Show frame details
        logger.info("\n📊 Extracted Frames:")
        for i, frame in enumerate(frames[:5]):
            logger.info(f"\n  Frame {i+1}:")
            logger.info(f"    File: {frame.path.name}")
            logger.info(f"    Time: {frame.timestamp:.1f}s")
            logger.info(f"    Reason: {frame.extraction_reason}")
            logger.info(f"    Quality: {frame.quality_score:.2f}")
            if frame.transcript_segment:
                logger.info(f"    Text: '{frame.transcript_segment.text}'")
        
        if len(frames) > 5:
            logger.info(f"\n  ... and {len(frames) - 5} more frames")
        
        logger.info("\n💾 Frames saved to: test_outputs/frames/")
        logger.info("   - Frame images (.jpg)")
        logger.info("   - metadata.json (frame details)")
        
        logger.info("\n🎯 What you can do with these frames:")
        logger.info("   1. View the extracted screenshots")
        logger.info("   2. See which moments were captured")
        logger.info("   3. Check metadata.json for frame-transcript links")
        logger.info("   4. Ready for Phase 3: Embeddings!")
        
    except Exception as e:
        logger.error(f"\n❌ Error during frame extraction: {e}")
        logger.info("\n💡 This is okay - frame extraction is optional")
        logger.info("   The transcript extraction still worked!")
    
    logger.info("\n" + "=" * 50)
    logger.success("✅ Complete Pipeline Test Finished!")
    logger.info("\n📚 What we tested:")
    logger.info("   ✓ Transcript extraction (audio → text)")
    logger.info("   ✓ Frame extraction (video → keyframes)")
    logger.info("   ✓ Transcript-frame alignment")
    logger.info("\n📁 Output files:")
    logger.info("   - test_outputs/transcript.json (transcript data)")
    logger.info("   - test_outputs/frames/ (extracted frames)")
    logger.info("\n🚀 Next: Try with your own tutorial videos!")
>>>>>>> e380357 (feat: Add frame extraction to test_it_yourself.py)


def show_usage_examples():
    """Show code examples of how to use the TranscriptExtractor"""
    
    logger.info("\n📖 Usage Examples:")
    logger.info("=" * 50)
    
    logger.info("\n1️⃣  Basic Usage:")
    logger.info("""
    from framewise import TranscriptExtractor
    
    extractor = TranscriptExtractor()
    transcript = extractor.extract("video.mp4")
    
    print(f"Language: {transcript.language}")
    print(f"Segments: {len(transcript.segments)}")
    """)
    
    logger.info("\n2️⃣  Save to JSON:")
    logger.info("""
    transcript = extractor.extract(
        "video.mp4",
        output_path="transcript.json"
    )
    """)
    
    logger.info("\n3️⃣  Batch Processing:")
    logger.info("""
    videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
    transcripts = extractor.extract_batch(
        videos,
        output_dir="transcripts/"
    )
    """)
    
    logger.info("\n4️⃣  Access Segments:")
    logger.info("""
    for segment in transcript.segments:
        print(f"[{segment.start}s]: {segment.text}")
    """)
    
    logger.info("\n5️⃣  Load Saved Transcript:")
    logger.info("""
    from framewise import Transcript
    
    transcript = Transcript.load("transcript.json")
    """)


if __name__ == "__main__":
    logger.info("🎬 FrameWise - Transcript Extractor Test")
    logger.info("=" * 50)
    
    # Show usage examples first
    show_usage_examples()
    
    logger.info("\n" + "=" * 50)
    logger.info("🧪 Running Test...")
    logger.info("=" * 50)
    
    # Run the test
    test_with_sample_video()

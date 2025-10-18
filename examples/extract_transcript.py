"""
Example: Extract transcript from a video file

This example shows how to use the TranscriptExtractor to generate
transcripts from tutorial videos.
"""

from pathlib import Path
from loguru import logger
from framewise.core.transcript_extractor import TranscriptExtractor, Transcript


def main():
    # Initialize the extractor
    # Use 'base' model for good balance of speed and accuracy
    # For faster processing, use 'tiny' or 'small'
    # For better accuracy, use 'medium' or 'large'
    extractor = TranscriptExtractor(
        model_size="base",
        language="en"  # Set to None for auto-detection
    )
    
    # Example 1: Extract transcript from a single video
    logger.info("Extracting transcript from video...")
    video_path = "path/to/your/tutorial.mp4"
    
    # Check if file exists (for demo purposes)
    if not Path(video_path).exists():
        logger.warning(f"Video file not found: {video_path}")
        logger.info("Please update the video_path variable with your video file")
        return
    
    transcript = extractor.extract(
        video_path=video_path,
        output_path="transcript.json"  # Save to JSON
    )
    
    logger.success("Transcript extracted!")
    logger.info(f"Language: {transcript.language}")
    logger.info(f"Segments: {len(transcript.segments)}")
    logger.info("\nFirst few segments:")
    for i, segment in enumerate(transcript.segments[:3]):
        logger.info(f"[{segment.start:.1f}s - {segment.end:.1f}s]: {segment.text}")
    
    logger.info(f"\nFull text preview:")
    logger.info(f"{transcript.full_text[:200]}...")
    
    # Example 2: Extract transcripts from multiple videos
    logger.info("="*50)
    logger.info("Batch processing example:")
    
    video_paths = [
        "path/to/tutorial1.mp4",
        "path/to/tutorial2.mp4",
        "path/to/tutorial3.mp4",
    ]
    
    # Filter to only existing files
    existing_videos = [v for v in video_paths if Path(v).exists()]
    
    if existing_videos:
        transcripts = extractor.extract_batch(
            video_paths=existing_videos,
            output_dir="transcripts/"  # Save all to this directory
        )
        
        logger.success(f"Processed {len(transcripts)} videos")
        for transcript in transcripts:
            logger.info(f"- {transcript.video_path.name}: {len(transcript.segments)} segments")
    else:
        logger.warning("No video files found for batch processing")
    
    # Example 3: Load a saved transcript
    if Path("transcript.json").exists():
        logger.info("="*50)
        logger.info("Loading saved transcript:")
        
        loaded_transcript = Transcript.load("transcript.json")
        logger.success(f"Loaded transcript for: {loaded_transcript.video_path.name}")
        logger.info(f"Total segments: {len(loaded_transcript.segments)}")


if __name__ == "__main__":
    main()

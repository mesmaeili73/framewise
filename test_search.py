"""
Quick test: Search for "how to open definely add-in"
"""

from pathlib import Path
from loguru import logger
from framewise import (
    TranscriptExtractor,
    FrameExtractor,
    FrameWiseEmbedder,
    FrameWiseVectorStore,
)


def main():
    VIDEO_PATH = "/Users/mes/Desktop/Screen Recording 2025-10-17 at 15.45.45.mov"
    
    logger.info("🎬 FrameWise Search Test")
    logger.info("=" * 60)
    
    # Step 1: Extract transcript
    logger.info("\n📝 Step 1: Extracting transcript...")
    transcript_extractor = TranscriptExtractor(model_size="base")
    transcript = transcript_extractor.extract(VIDEO_PATH)
    logger.success(f"✓ {len(transcript.segments)} segments")
    
    # Step 2: Extract frames
    logger.info("\n🖼️  Step 2: Extracting frames...")
    frame_extractor = FrameExtractor(
        strategy="hybrid",
        max_frames_per_video=15,
        scene_threshold=0.3,
        quality_threshold=0.1
    )
    frames = frame_extractor.extract(
        video_path=VIDEO_PATH,
        transcript=transcript,
        output_dir="test_outputs/frames"
    )
    logger.success(f"✓ {len(frames)} frames")
    
    if len(frames) == 0:
        logger.warning("No frames extracted - cannot test search")
        logger.info("Try adjusting quality_threshold or scene_threshold")
        return
    
    # Step 3: Generate embeddings
    logger.info("\n🧠 Step 3: Generating embeddings...")
    embedder = FrameWiseEmbedder()
    embeddings = embedder.embed_frames_batch(frames, batch_size=4)
    logger.success(f"✓ {len(embeddings)} embeddings")
    
    # Step 4: Store in database
    logger.info("\n💾 Step 4: Creating vector database...")
    vector_store = FrameWiseVectorStore(
        db_path="test_outputs/search_test.db",
        table_name="frames"
    )
    vector_store.create_table(embeddings, mode="overwrite")
    logger.success("✓ Database created")
    
    # Step 5: Search!
    logger.info("\n🔍 Step 5: Searching...")
    logger.info("=" * 60)
    
    query = "how to open definely add-in"
    logger.info(f"\n🔎 Query: '{query}'")
    
    results = vector_store.search_by_text(
        query_text=query,
        embedder=embedder,
        limit=5,
        search_type="hybrid"
    )
    
    if results:
        logger.success(f"\n✅ Found {len(results)} relevant frames:")
        for i, result in enumerate(results, 1):
            logger.info(f"\n📍 Result {i}:")
            logger.info(f"   Timestamp: {result['timestamp']:.1f}s")
            logger.info(f"   Text: '{result['text']}'")
            logger.info(f"   Frame: {Path(result['frame_path']).name}")
            logger.info(f"   Distance: {result.get('_distance', 'N/A')}")
    else:
        logger.warning("No results found")
    
    logger.info("\n" + "=" * 60)
    logger.success("✅ Search test complete!")


if __name__ == "__main__":
    main()

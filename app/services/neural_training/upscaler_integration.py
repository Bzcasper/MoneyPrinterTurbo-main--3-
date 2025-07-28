"""
AI Upscaler Integration Layer for MoneyPrinterTurbo
==================================================

Integration layer that connects the AI upscaler with the main video processing pipeline.
Provides seamless integration with existing video enhancement workflows.

Author: AIUpscaler Agent
Version: 1.0.0
"""

import os
import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import tempfile

from loguru import logger

# Import main components
from app.services.neural_training.ai_upscaler import (
    AIVideoUpscaler, UpscalerConfig, upscale_video_ai,
    create_fast_upscaler, create_balanced_upscaler, create_ultra_upscaler
)
from app.services.quality_enhancement import VideoQualityEnhancer, QualityEnhancementConfig
from app.services.video import MemoryMonitor
from app.models.schema import VideoParams


class UpscalerIntegrationManager:
    """Manager for integrating AI upscaler with video processing pipeline"""
    
    def __init__(self):
        self.upscaler = None
        self.quality_enhancer = None
        self.processing_stats = {
            'total_videos_processed': 0,
            'total_processing_time': 0,
            'average_scale_factor': 0,
            'success_rate': 0.0
        }
        
        logger.info("AI Upscaler Integration Manager initialized")
    
    async def initialize(self, quality_preset: str = "balanced"):
        """Initialize the upscaler and quality enhancer"""
        logger.info(f"🔄 Initializing AI upscaling system (preset: {quality_preset})")
        
        try:
            # Create upscaler based on preset
            if quality_preset == "fast":
                self.upscaler = create_fast_upscaler()
            elif quality_preset == "ultra":
                self.upscaler = create_ultra_upscaler()
            else:
                self.upscaler = create_balanced_upscaler()
            
            # Initialize upscaler models
            await self.upscaler.initialize_models()
            
            # Initialize quality enhancer
            quality_config = QualityEnhancementConfig()
            quality_config.neural_upscaling = True
            quality_config.neural_quality_enhancement = True
            self.quality_enhancer = VideoQualityEnhancer(quality_config)
            
            logger.success("✅ AI upscaling system initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI upscaling system: {e}")
            raise
    
    async def process_video_with_upscaling(
        self,
        input_path: str,
        output_path: str,
        params: VideoParams,
        scale_factor: int = 4,
        apply_quality_enhancement: bool = True
    ) -> Dict[str, Any]:
        """
        Process video with AI upscaling and optional quality enhancement
        
        Args:
            input_path: Path to input video
            output_path: Path for final output
            params: Video processing parameters
            scale_factor: Upscaling factor (2, 4, or 8)
            apply_quality_enhancement: Whether to apply additional quality enhancements
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        logger.info(f"🎬 Processing video with {scale_factor}x AI upscaling")
        logger.info(f"📹 Input: {input_path}")
        logger.info(f"💾 Output: {output_path}")
        
        try:
            # Ensure system is initialized
            if not self.upscaler:
                await self.initialize()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Step 1: AI Upscaling
                upscaled_path = os.path.join(temp_dir, "upscaled_video.mp4")
                
                logger.info("🚀 Starting AI upscaling...")
                upscaling_result = await self.upscaler.upscale_video(
                    input_path, upscaled_path, scale_factor
                )
                
                if not upscaling_result['success']:
                    raise RuntimeError(f"Upscaling failed: {upscaling_result.get('error')}")
                
                logger.success(f"✅ AI upscaling completed in {upscaling_result['processing_time']:.2f}s")
                
                # Step 2: Optional Quality Enhancement
                final_path = output_path
                if apply_quality_enhancement and self.quality_enhancer:
                    logger.info("🎨 Applying additional quality enhancements...")
                    
                    enhancement_result = await self.quality_enhancer.enhance_video(
                        upscaled_path, output_path, params
                    )
                    
                    if enhancement_result['success']:
                        logger.success("✅ Quality enhancement completed")
                    else:
                        logger.warning("⚠️ Quality enhancement failed, using upscaled video")
                        import shutil
                        shutil.copy2(upscaled_path, output_path)
                else:
                    # Just copy upscaled video to final output
                    import shutil
                    shutil.copy2(upscaled_path, output_path)
                
                # Step 3: Coordinate with other agents
                await self._coordinate_with_agents({
                    'input_path': input_path,
                    'output_path': output_path,
                    'scale_factor': scale_factor,
                    'output_resolution': upscaling_result['output_resolution'],
                    'processing_time': upscaling_result['processing_time']
                })
                
                # Update statistics
                total_time = time.time() - start_time
                self._update_processing_stats(scale_factor, total_time, True)
                
                logger.success(f"🎉 Video processing completed in {total_time:.2f}s")
                
                return {
                    'success': True,
                    'output_path': final_path,
                    'scale_factor': scale_factor,
                    'original_resolution': (
                        upscaling_result['output_resolution'][0] // scale_factor,
                        upscaling_result['output_resolution'][1] // scale_factor
                    ),
                    'upscaled_resolution': upscaling_result['output_resolution'],
                    'upscaling_time': upscaling_result['processing_time'],
                    'total_processing_time': total_time,
                    'frames_processed': upscaling_result['frames_processed'],
                    'average_fps': upscaling_result['average_fps'],
                    'quality_enhanced': apply_quality_enhancement
                }
                
        except Exception as e:
            logger.error(f"❌ Video processing failed: {str(e)}")
            total_time = time.time() - start_time
            self._update_processing_stats(scale_factor, total_time, False)
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': total_time
            }
    
    async def _coordinate_with_agents(self, processing_data: Dict[str, Any]):
        """Coordinate with other agents in the swarm"""
        logger.info("🔗 Coordinating with other agents...")
        
        try:
            # Coordinate with FrameInterpolator if needed
            if self.upscaler:
                await self.upscaler.coordinate_with_frame_interpolator(processing_data)
            
            # Store processing results in swarm memory
            memory_data = {
                'timestamp': time.time(),
                'agent': 'AIUpscaler',
                'processing_data': processing_data,
                'quality_metrics': self.processing_stats
            }
            
            # Use claude-flow hooks to notify other agents
            import subprocess
            result = subprocess.run([
                'npx', 'claude-flow@alpha', 'hooks', 'notification',
                '--message', 'AI upscaling completed',
                '--level', 'completion',
                '--data', json.dumps(memory_data)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.debug("🔔 Agent coordination completed")
            else:
                logger.warning("⚠️ Agent coordination had issues")
                
        except Exception as e:
            logger.warning(f"⚠️ Agent coordination error: {e}")
    
    def _update_processing_stats(self, scale_factor: int, processing_time: float, success: bool):
        """Update processing statistics"""
        self.processing_stats['total_videos_processed'] += 1
        self.processing_stats['total_processing_time'] += processing_time
        
        # Update average scale factor
        current_avg = self.processing_stats['average_scale_factor']
        count = self.processing_stats['total_videos_processed']
        self.processing_stats['average_scale_factor'] = (
            (current_avg * (count - 1) + scale_factor) / count
        )
        
        # Update success rate
        if success:
            success_count = self.processing_stats['success_rate'] * (count - 1) + 1
        else:
            success_count = self.processing_stats['success_rate'] * (count - 1)
        
        self.processing_stats['success_rate'] = success_count / count
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()
    
    async def batch_process_videos(
        self,
        input_paths: List[str],
        output_dir: str,
        scale_factor: int = 4,
        quality_preset: str = "balanced"
    ) -> List[Dict[str, Any]]:
        """
        Process multiple videos in batch
        
        Args:
            input_paths: List of input video paths
            output_dir: Directory for output videos
            scale_factor: Upscaling factor
            quality_preset: Quality preset to use
            
        Returns:
            List of processing results
        """
        logger.info(f"📦 Starting batch processing of {len(input_paths)} videos")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize with specified preset
        await self.initialize(quality_preset)
        
        results = []
        
        for i, input_path in enumerate(input_paths):
            logger.info(f"📹 Processing video {i+1}/{len(input_paths)}: {input_path}")
            
            # Generate output path
            input_name = Path(input_path).stem
            output_path = os.path.join(output_dir, f"{input_name}_upscaled_{scale_factor}x.mp4")
            
            # Create dummy VideoParams (you may want to customize this)
            params = VideoParams(
                video_subject="batch_processing",
                voice_name="en-US-JennyNeural",
                subtitle_enabled=False
            )
            
            # Process video
            result = await self.process_video_with_upscaling(
                input_path, output_path, params, scale_factor
            )
            
            result['input_path'] = input_path
            result['video_index'] = i + 1
            results.append(result)
            
            # Log progress
            if result['success']:
                logger.success(f"✅ Video {i+1} completed: {output_path}")
            else:
                logger.error(f"❌ Video {i+1} failed: {result.get('error')}")
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        logger.info(f"📊 Batch processing completed: {successful}/{len(input_paths)} successful")
        
        return results


# Global instance for easy access
_integration_manager = None


async def get_upscaler_manager() -> UpscalerIntegrationManager:
    """Get or create the global upscaler integration manager"""
    global _integration_manager
    
    if _integration_manager is None:
        _integration_manager = UpscalerIntegrationManager()
    
    return _integration_manager


# Convenience functions for common operations

async def upscale_video_integrated(
    input_path: str,
    output_path: str,
    params: VideoParams,
    scale_factor: int = 4,
    quality_preset: str = "balanced"
) -> Dict[str, Any]:
    """
    Convenience function for integrated video upscaling
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        params: Video processing parameters
        scale_factor: Upscaling factor (2, 4, or 8)
        quality_preset: Quality preset ("fast", "balanced", "ultra")
        
    Returns:
        Dictionary with processing results
    """
    manager = await get_upscaler_manager()
    await manager.initialize(quality_preset)
    
    return await manager.process_video_with_upscaling(
        input_path, output_path, params, scale_factor
    )


async def batch_upscale_videos(
    input_paths: List[str],
    output_dir: str,
    scale_factor: int = 4,
    quality_preset: str = "balanced"
) -> List[Dict[str, Any]]:
    """
    Convenience function for batch video upscaling
    
    Args:
        input_paths: List of input video paths
        output_dir: Directory for output videos
        scale_factor: Upscaling factor
        quality_preset: Quality preset
        
    Returns:
        List of processing results
    """
    manager = await get_upscaler_manager()
    
    return await manager.batch_process_videos(
        input_paths, output_dir, scale_factor, quality_preset
    )


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Video Upscaler Integration")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor (2, 4, 8)")
    parser.add_argument("--preset", default="balanced", help="Quality preset (fast, balanced, ultra)")
    parser.add_argument("--batch", help="Batch mode: input directory")
    parser.add_argument("--output-dir", help="Output directory for batch mode")
    
    args = parser.parse_args()
    
    async def main():
        if args.batch:
            # Batch processing mode
            input_paths = list(Path(args.batch).glob("*.mp4"))
            output_dir = args.output_dir or "upscaled_videos"
            
            results = await batch_upscale_videos(
                [str(p) for p in input_paths],
                output_dir,
                args.scale,
                args.preset
            )
            
            print(f"Batch processing completed: {len(results)} videos processed")
            
        else:
            # Single video processing
            params = VideoParams(
                video_subject="test_upscaling",
                voice_name="en-US-JennyNeural",
                subtitle_enabled=False
            )
            
            result = await upscale_video_integrated(
                args.input,
                args.output,
                params,
                args.scale,
                args.preset
            )
            
            if result['success']:
                print(f"Upscaling successful: {result['output_path']}")
                print(f"Processing time: {result['total_processing_time']:.2f}s")
                print(f"Resolution: {result['original_resolution']} → {result['upscaled_resolution']}")
            else:
                print(f"Upscaling failed: {result['error']}")
    
    asyncio.run(main())
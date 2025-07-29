"""
Video Processing Workflow Integration

Integrates the comprehensive workflow monitoring system with the existing
video processing pipeline to provide real-time progress tracking and
performance monitoring for video generation tasks.

This module demonstrates how to integrate workflow monitoring into existing
MoneyPrinterTurbo video processing operations.
"""

import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from loguru import logger

from app.services.workflow_monitor import (
    workflow_monitor,
    create_workflow,
    add_step,
    start_workflow,
    start_step,
    update_progress,
    complete_step,
    fail_step,
    record_metric,
    get_status
)
from app.services import task as video_task
from app.models.schema import VideoParams


class VideoWorkflowMonitor:
    """
    Integration wrapper for video processing with workflow monitoring
    
    Provides seamless integration between the existing video processing
    pipeline and the new workflow monitoring system.
    """
    
    def __init__(self):
        """Initialize video workflow monitor"""
        self.workflow_id: Optional[str] = None
        self.step_mapping: Dict[str, str] = {}
        
        # Ensure monitoring is started
        workflow_monitor.start_monitoring()
        
        logger.info("VideoWorkflowMonitor initialized")
    
    def create_video_workflow(self, task_id: str, params: VideoParams) -> str:
        """Create a monitored video processing workflow"""
        
        # Create workflow
        workflow_name = f"Video Generation - {params.video_subject[:50]}"
        workflow_description = f"Generate video for task {task_id} with {params.video_count} output(s)"
        
        self.workflow_id = create_workflow(workflow_name, workflow_description)
        
        # Define video processing steps based on the existing pipeline
        video_steps = [
            ("script_generation", "Generate Video Script", "Create engaging video script content"),
            ("terms_generation", "Generate Search Terms", "Generate terms for video material search"),
            ("audio_generation", "Generate Audio", "Convert script to speech audio"),
            ("subtitle_generation", "Generate Subtitles", "Create subtitle files for video"),
            ("material_download", "Download Materials", "Download video materials from sources"),
            ("video_processing", "Process Videos", "Combine materials and generate final videos")
        ]
        
        # Add steps to workflow
        for step_key, step_name, step_desc in video_steps:
            step_id = add_step(self.workflow_id, step_name, step_desc)
            self.step_mapping[step_key] = step_id
        
        logger.info(f"Created video workflow {self.workflow_id} for task {task_id}")
        
        # Record initial metrics
        record_metric(self.workflow_id, "video_count", params.video_count, "videos")
        record_metric(self.workflow_id, "paragraph_count", params.paragraph_number, "paragraphs")
        record_metric(self.workflow_id, "voice_rate", params.voice_rate, "rate")
        
        return self.workflow_id
    
    def start_video_processing(self) -> None:
        """Start the video processing workflow"""
        if not self.workflow_id:
            raise ValueError("No workflow created. Call create_video_workflow first.")
        
        start_workflow(self.workflow_id)
        logger.info(f"Started video workflow {self.workflow_id}")
    
    def execute_script_generation(self, task_id: str, params: VideoParams) -> Optional[str]:
        """Execute script generation with monitoring"""
        step_id = self.step_mapping["script_generation"]
        
        try:
            # Start step
            start_step(self.workflow_id, step_id)
            update_progress(self.workflow_id, step_id, 0, "Initializing script generation...")
            
            start_time = time.time()
            
            # Call original function
            update_progress(self.workflow_id, step_id, 25, "Generating script content...")
            video_script = video_task.generate_script(task_id, params)
            
            if not video_script or "Error: " in video_script:
                fail_step(self.workflow_id, step_id, "Failed to generate video script")
                return None
            
            # Record metrics
            execution_time = time.time() - start_time
            record_metric(self.workflow_id, "script_generation_time", execution_time, "seconds", step_id)
            record_metric(self.workflow_id, "script_length", len(video_script), "characters", step_id)
            
            # Complete step
            update_progress(self.workflow_id, step_id, 100, "Script generation completed")
            complete_step(self.workflow_id, step_id, {
                "script_length": len(video_script),
                "execution_time": execution_time
            })
            
            logger.info(f"Script generation completed in {execution_time:.2f}s")
            return video_script
            
        except Exception as e:
            error_msg = f"Script generation failed: {str(e)}"
            fail_step(self.workflow_id, step_id, error_msg)
            logger.error(error_msg)
            return None
    
    def execute_terms_generation(self, task_id: str, params: VideoParams, video_script: str) -> Optional[List[str]]:
        """Execute terms generation with monitoring"""
        step_id = self.step_mapping["terms_generation"]
        
        try:
            # Start step
            start_step(self.workflow_id, step_id)
            update_progress(self.workflow_id, step_id, 0, "Starting terms generation...")
            
            start_time = time.time()
            
            # Call original function
            update_progress(self.workflow_id, step_id, 50, "Generating search terms...")
            video_terms = video_task.generate_terms(task_id, params, video_script)
            
            if not video_terms:
                fail_step(self.workflow_id, step_id, "Failed to generate video terms")
                return None
            
            # Record metrics
            execution_time = time.time() - start_time
            record_metric(self.workflow_id, "terms_generation_time", execution_time, "seconds", step_id)
            record_metric(self.workflow_id, "terms_count", len(video_terms), "terms", step_id)
            
            # Complete step
            update_progress(self.workflow_id, step_id, 100, f"Generated {len(video_terms)} search terms")
            complete_step(self.workflow_id, step_id, {
                "terms_count": len(video_terms),
                "execution_time": execution_time,
                "terms": video_terms
            })
            
            logger.info(f"Terms generation completed: {len(video_terms)} terms in {execution_time:.2f}s")
            return video_terms
            
        except Exception as e:
            error_msg = f"Terms generation failed: {str(e)}"
            fail_step(self.workflow_id, step_id, error_msg)
            logger.error(error_msg)
            return None
    
    def execute_audio_generation(self, task_id: str, params: VideoParams, video_script: str) -> Optional[tuple]:
        """Execute audio generation with monitoring"""
        step_id = self.step_mapping["audio_generation"]
        
        try:
            # Start step
            start_step(self.workflow_id, step_id)
            update_progress(self.workflow_id, step_id, 0, "Initializing TTS engine...")
            
            start_time = time.time()
            
            # Call original function with progress updates
            update_progress(self.workflow_id, step_id, 25, "Converting text to speech...")
            audio_result = video_task.generate_audio(task_id, params, video_script)
            
            if not audio_result or not audio_result[0]:
                fail_step(self.workflow_id, step_id, "Failed to generate audio")
                return None
            
            audio_file, audio_duration, sub_maker = audio_result
            
            # Record metrics
            execution_time = time.time() - start_time
            record_metric(self.workflow_id, "audio_generation_time", execution_time, "seconds", step_id)
            record_metric(self.workflow_id, "audio_duration", audio_duration, "seconds", step_id)
            
            # Check audio file size
            audio_path = Path(audio_file)
            if audio_path.exists():
                audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
                record_metric(self.workflow_id, "audio_file_size", audio_size_mb, "MB", step_id)
            
            # Complete step
            update_progress(self.workflow_id, step_id, 100, f"Audio generated: {audio_duration}s duration")
            complete_step(self.workflow_id, step_id, {
                "audio_duration": audio_duration,
                "execution_time": execution_time,
                "audio_file": audio_file
            })
            
            logger.info(f"Audio generation completed: {audio_duration}s in {execution_time:.2f}s")
            return audio_result
            
        except Exception as e:
            error_msg = f"Audio generation failed: {str(e)}"
            fail_step(self.workflow_id, step_id, error_msg)
            logger.error(error_msg)
            return None
    
    def execute_subtitle_generation(self, task_id: str, params: VideoParams, video_script: str, 
                                  sub_maker: Any, audio_file: str) -> Optional[str]:
        """Execute subtitle generation with monitoring"""
        step_id = self.step_mapping["subtitle_generation"]
        
        try:
            # Start step
            start_step(self.workflow_id, step_id)
            update_progress(self.workflow_id, step_id, 0, "Starting subtitle generation...")
            
            start_time = time.time()
            
            # Call original function
            update_progress(self.workflow_id, step_id, 30, "Processing subtitle timing...")
            subtitle_path = video_task.generate_subtitle(task_id, params, video_script, sub_maker, audio_file)
            
            # Record metrics
            execution_time = time.time() - start_time
            record_metric(self.workflow_id, "subtitle_generation_time", execution_time, "seconds", step_id)
            
            if subtitle_path:
                # Check subtitle file
                subtitle_file = Path(subtitle_path)
                if subtitle_file.exists():
                    subtitle_size_kb = subtitle_file.stat().st_size / 1024
                    record_metric(self.workflow_id, "subtitle_file_size", subtitle_size_kb, "KB", step_id)
                    
                    # Count subtitle lines
                    with open(subtitle_path, 'r', encoding='utf-8') as f:
                        lines = len([line for line in f if line.strip()])
                    record_metric(self.workflow_id, "subtitle_lines", lines, "lines", step_id)
            
            # Complete step
            update_progress(self.workflow_id, step_id, 100, 
                          "Subtitles generated" if subtitle_path else "Subtitles skipped")
            complete_step(self.workflow_id, step_id, {
                "subtitle_path": subtitle_path,
                "execution_time": execution_time,
                "subtitles_enabled": params.subtitle_enabled
            })
            
            logger.info(f"Subtitle generation completed in {execution_time:.2f}s")
            return subtitle_path
            
        except Exception as e:
            error_msg = f"Subtitle generation failed: {str(e)}"
            fail_step(self.workflow_id, step_id, error_msg)
            logger.error(error_msg)
            return None
    
    def execute_material_download(self, task_id: str, params: VideoParams, 
                                video_terms: List[str], audio_duration: float) -> Optional[List[str]]:
        """Execute material download with monitoring"""
        step_id = self.step_mapping["material_download"]
        
        try:
            # Start step
            start_step(self.workflow_id, step_id)
            update_progress(self.workflow_id, step_id, 0, "Initializing material download...")
            
            start_time = time.time()
            
            # Call original function
            update_progress(self.workflow_id, step_id, 20, "Searching for video materials...")
            downloaded_videos = video_task.get_video_materials(task_id, params, video_terms, audio_duration)
            
            if not downloaded_videos:
                fail_step(self.workflow_id, step_id, "Failed to download video materials")
                return None
            
            # Record metrics
            execution_time = time.time() - start_time
            record_metric(self.workflow_id, "material_download_time", execution_time, "seconds", step_id)
            record_metric(self.workflow_id, "materials_count", len(downloaded_videos), "files", step_id)
            
            # Calculate total file size
            total_size_mb = 0
            for video_path in downloaded_videos:
                video_file = Path(video_path)
                if video_file.exists():
                    total_size_mb += video_file.stat().st_size / (1024 * 1024)
            
            record_metric(self.workflow_id, "materials_total_size", total_size_mb, "MB", step_id)
            
            # Complete step
            update_progress(self.workflow_id, step_id, 100, 
                          f"Downloaded {len(downloaded_videos)} materials ({total_size_mb:.1f}MB)")
            complete_step(self.workflow_id, step_id, {
                "materials_count": len(downloaded_videos),
                "total_size_mb": total_size_mb,
                "execution_time": execution_time
            })
            
            logger.info(f"Material download completed: {len(downloaded_videos)} files in {execution_time:.2f}s")
            return downloaded_videos
            
        except Exception as e:
            error_msg = f"Material download failed: {str(e)}"
            fail_step(self.workflow_id, step_id, error_msg)
            logger.error(error_msg)
            return None
    
    def execute_video_processing(self, task_id: str, params: VideoParams, 
                               downloaded_videos: List[str], audio_file: str, 
                               subtitle_path: str) -> Optional[tuple]:
        """Execute final video processing with monitoring"""
        step_id = self.step_mapping["video_processing"]
        
        try:
            # Start step
            start_step(self.workflow_id, step_id)
            update_progress(self.workflow_id, step_id, 0, "Starting video processing...")
            
            start_time = time.time()
            
            # Call original function with progress monitoring
            update_progress(self.workflow_id, step_id, 25, "Combining video materials...")
            
            # Hook into the progress updates from the original video processing
            self._monitor_video_generation_progress(step_id, params.video_count)
            
            final_video_paths, combined_video_paths = video_task.generate_final_videos(
                task_id, params, downloaded_videos, audio_file, subtitle_path
            )
            
            if not final_video_paths:
                fail_step(self.workflow_id, step_id, "Failed to generate final videos")
                return None
            
            # Record metrics
            execution_time = time.time() - start_time
            record_metric(self.workflow_id, "video_processing_time", execution_time, "seconds", step_id)
            record_metric(self.workflow_id, "final_videos_count", len(final_video_paths), "videos", step_id)
            
            # Calculate output video sizes
            total_output_size_mb = 0
            for video_path in final_video_paths:
                video_file = Path(video_path)
                if video_file.exists():
                    total_output_size_mb += video_file.stat().st_size / (1024 * 1024)
            
            record_metric(self.workflow_id, "output_videos_size", total_output_size_mb, "MB", step_id)
            
            # Complete step
            update_progress(self.workflow_id, step_id, 100, 
                          f"Generated {len(final_video_paths)} videos ({total_output_size_mb:.1f}MB)")
            complete_step(self.workflow_id, step_id, {
                "final_videos_count": len(final_video_paths),
                "output_size_mb": total_output_size_mb,
                "execution_time": execution_time,
                "video_paths": final_video_paths
            })
            
            logger.info(f"Video processing completed: {len(final_video_paths)} videos in {execution_time:.2f}s")
            return final_video_paths, combined_video_paths
            
        except Exception as e:
            error_msg = f"Video processing failed: {str(e)}"
            fail_step(self.workflow_id, step_id, error_msg)
            logger.error(error_msg)
            return None
    
    def _monitor_video_generation_progress(self, step_id: str, video_count: int) -> None:
        """Monitor progress during video generation"""
        # This would ideally hook into the existing video generation progress
        # For now, simulate progress updates
        
        progress_points = [30, 45, 60, 75, 90]
        messages = [
            "Processing video effects...",
            "Encoding video streams...",
            "Applying audio sync...",
            "Optimizing output quality...",
            "Finalizing video files..."
        ]
        
        for progress, message in zip(progress_points, messages):
            update_progress(self.workflow_id, step_id, progress, message)
            time.sleep(0.1)  # Small delay to simulate work
    
    def get_workflow_status(self) -> Optional[Dict[str, Any]]:
        """Get current workflow status"""
        if not self.workflow_id:
            return None
        return get_status(self.workflow_id)
    
    def get_workflow_report(self) -> Optional[str]:
        """Generate workflow report"""
        if not self.workflow_id:
            return None
        
        from app.services.workflow_monitor import generate_report
        return generate_report(self.workflow_id)
    
    def execute_complete_workflow(self, task_id: str, params: VideoParams, stop_at: str = "video") -> Optional[Dict[str, Any]]:
        """Execute complete video workflow with monitoring"""
        
        try:
            # Create and start workflow
            self.create_video_workflow(task_id, params)
            self.start_video_processing()
            
            logger.info(f"Starting complete video workflow for task {task_id}")
            
            # Execute steps based on stop_at parameter
            
            # 1. Generate script
            video_script = self.execute_script_generation(task_id, params)
            if not video_script:
                return None
            
            if stop_at == "script":
                return {"script": video_script}
            
            # 2. Generate terms (if not using local materials)
            video_terms = []
            if params.video_source != "local":
                video_terms = self.execute_terms_generation(task_id, params, video_script)
                if not video_terms:
                    return None
            
            if stop_at == "terms":
                return {"script": video_script, "terms": video_terms}
            
            # 3. Generate audio
            audio_result = self.execute_audio_generation(task_id, params, video_script)
            if not audio_result:
                return None
            
            audio_file, audio_duration, sub_maker = audio_result
            
            if stop_at == "audio":
                return {"audio_file": audio_file, "audio_duration": audio_duration}
            
            # 4. Generate subtitles
            subtitle_path = self.execute_subtitle_generation(
                task_id, params, video_script, sub_maker, audio_file
            )
            
            if stop_at == "subtitle":
                return {"subtitle_path": subtitle_path}
            
            # 5. Download materials
            downloaded_videos = self.execute_material_download(
                task_id, params, video_terms, audio_duration
            )
            if not downloaded_videos:
                return None
            
            if stop_at == "materials":
                return {"materials": downloaded_videos}
            
            # 6. Process final videos
            video_result = self.execute_video_processing(
                task_id, params, downloaded_videos, audio_file, subtitle_path
            )
            if not video_result:
                return None
            
            final_video_paths, combined_video_paths = video_result
            
            # Return complete results
            result = {
                "videos": final_video_paths,
                "combined_videos": combined_video_paths,
                "script": video_script,
                "terms": video_terms,
                "audio_file": audio_file,
                "audio_duration": audio_duration,
                "subtitle_path": subtitle_path,
                "materials": downloaded_videos,
                "workflow_id": self.workflow_id
            }
            
            logger.success(f"Video workflow completed successfully: {len(final_video_paths)} videos generated")
            return result
            
        except Exception as e:
            error_msg = f"Video workflow failed: {str(e)}"
            logger.error(error_msg)
            
            # If we have a workflow, mark it as failed
            if self.workflow_id:
                # Create alert for workflow failure
                from app.services.workflow_monitor import workflow_monitor
                workflow_monitor._create_alert(
                    workflow_id=self.workflow_id,
                    severity=workflow_monitor.AlertSeverity.CRITICAL,
                    message=error_msg
                )
            
            return None


# Convenience function to replace the original task.start function
def start_monitored_video_task(task_id: str, params: VideoParams, stop_at: str = "video") -> Optional[Dict[str, Any]]:
    """
    Start a video generation task with comprehensive workflow monitoring
    
    This function replaces the original task.start function and provides
    the same interface while adding comprehensive monitoring capabilities.
    
    Args:
        task_id: Unique task identifier
        params: Video generation parameters
        stop_at: Where to stop in the workflow (script, terms, audio, subtitle, materials, video)
    
    Returns:
        Dictionary with results or None if failed
    """
    
    monitor = VideoWorkflowMonitor()
    return monitor.execute_complete_workflow(task_id, params, stop_at)


# Example usage and integration demonstration
def demonstrate_video_workflow_monitoring():
    """Demonstrate video workflow monitoring integration"""
    
    print("🎬 Video Workflow Monitoring Demonstration")
    print("="*60)
    
    # Create sample parameters
    from app.models.schema import VideoParams, VideoConcatMode
    
    params = VideoParams(
        video_subject="The Future of AI Technology",
        video_language="en",
        paragraph_number=3,
        voice_name="en-US-JennyNeural-Female",
        voice_rate=1.0,
        video_count=1,
        video_source="pexels",
        subtitle_enabled=True,
        video_concat_mode=VideoConcatMode.random
    )
    
    # Execute monitored workflow
    task_id = f"demo_task_{int(time.time())}"
    
    monitor = VideoWorkflowMonitor()
    
    # Create workflow
    workflow_id = monitor.create_video_workflow(task_id, params)
    print(f"📋 Created workflow: {workflow_id}")
    
    # Show initial status
    status = monitor.get_workflow_status()
    if status:
        print(f"📊 Initial progress: {status['progress']:.1f}%")
        print(f"📦 Total steps: {status['total_steps']}")
    
    print("\n🚀 Starting video processing with monitoring...")
    
    # Note: In a real scenario, you would call:
    # result = monitor.execute_complete_workflow(task_id, params)
    
    # For demonstration, we'll simulate the workflow
    monitor.start_video_processing()
    
    # Simulate script generation
    start_step(workflow_id, monitor.step_mapping["script_generation"])
    update_progress(workflow_id, monitor.step_mapping["script_generation"], 100, "Script completed")
    complete_step(workflow_id, monitor.step_mapping["script_generation"], {"success": True})
    
    print("✅ Script generation completed")
    
    # Show updated status
    status = monitor.get_workflow_status()
    if status:
        print(f"📊 Current progress: {status['progress']:.1f}%")
        print(f"✅ Completed steps: {status['completed_steps']}/{status['total_steps']}")
    
    # Generate report
    report = monitor.get_workflow_report()
    if report:
        print("\n📊 Workflow Report:")
        print(report)
    
    print("\n🎉 Video workflow monitoring demonstration complete!")


if __name__ == "__main__":
    demonstrate_video_workflow_monitoring()
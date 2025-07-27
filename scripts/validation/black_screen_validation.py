#!/usr/bin/env python3
"""
BLACK SCREEN BUG VALIDATION TEST
Critical Focus: Single Clip Processing Fix

Integration Tester Agent - Hive Mind Swarm
Agent ID: agent_1753116047042_3mdgua
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).parent))

from app.models.schema import MaterialInfo
from app.services import video as video_service
from moviepy import VideoFileClip
import psutil

def test_black_screen_bug_fix():
    """Critical test: Validate that single clip processing doesn't produce black screens"""
    print("="*60)
    print("BLACK SCREEN BUG VALIDATION TEST")
    print("Critical Path: Single Clip Processing")
    print("="*60)
    
    test_resources_dir = "test/resources"
    if not os.path.exists(test_resources_dir):
        print(f"❌ Test resources directory not found: {test_resources_dir}")
        return False
    
    # Find available test images
    test_images = []
    for i in range(1, 10):
        img_path = os.path.join(test_resources_dir, f"{i}.png")
        if os.path.exists(img_path):
            test_images.append(img_path)
    
    print(f"Found {len(test_images)} test images")
    
    if not test_images:
        print("❌ No test images found. Run create_test_images.py first.")
        return False
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Single Clip Processing (THE CRITICAL TEST)
    print("\n🔥 CRITICAL TEST: Single Clip Black Screen Bug Fix")
    print("-" * 50)
    
    for i, img_path in enumerate(test_images[:3]):  # Test first 3 images
        print(f"\nTesting image {i+1}: {os.path.basename(img_path)}")
        
        try:
            # Create material
            material = MaterialInfo()
            material.url = img_path
            material.provider = "local"
            material.duration = 0
            
            # Test different durations
            for duration in [1, 2, 3, 5]:
                total_tests += 1
                print(f"  Duration: {duration}s... ", end="", flush=True)
                
                start_time = time.time()
                
                # THE CRITICAL CALL - This previously caused black screens
                result = video_service.preprocess_video([material], clip_duration=duration)
                
                process_time = time.time() - start_time
                
                # Validate result
                if not result or len(result) != 1:
                    print(f"❌ FAILED - Invalid result")
                    continue
                
                output_path = result[0].url
                if not os.path.exists(output_path):
                    print(f"❌ FAILED - Output file doesn't exist")
                    continue
                
                file_size = os.path.getsize(output_path)
                if file_size < 1000:  # Very small files are likely empty/corrupt
                    print(f"❌ FAILED - Output too small ({file_size} bytes)")
                    os.remove(output_path)
                    continue
                
                # Validate video properties
                try:
                    clip = VideoFileClip(output_path)
                    actual_duration = clip.duration
                    frame_count = int(clip.fps * clip.duration) if clip.fps else 0
                    
                    # Check for black screen indicators
                    black_screen = False
                    if frame_count == 0:
                        black_screen = True
                    elif actual_duration <= 0:
                        black_screen = True
                    
                    clip.close()
                    
                    if black_screen:
                        print(f"❌ FAILED - Black screen detected (duration: {actual_duration}s)")
                        os.remove(output_path)
                        continue
                    
                    # Success!
                    success_count += 1
                    print(f"✅ PASSED ({process_time:.2f}s, {file_size} bytes, {actual_duration:.1f}s duration)")
                    
                    # Cleanup
                    os.remove(output_path)
                    
                except Exception as e:
                    print(f"❌ FAILED - Video validation error: {e}")
                    if os.path.exists(output_path):
                        os.remove(output_path)
        
        except Exception as e:
            print(f"❌ CRITICAL ERROR processing {img_path}: {e}")
    
    # Test 2: Multi-clip processing
    print(f"\n🔥 MULTI-CLIP TEST: Multiple clips processing")
    print("-" * 50)
    
    if len(test_images) >= 3:
        try:
            materials = []
            for i in range(3):
                material = MaterialInfo()
                material.url = test_images[i]
                material.provider = "local"
                material.duration = 0
                materials.append(material)
            
            total_tests += 1
            print(f"Processing {len(materials)} clips... ", end="", flush=True)
            
            start_time = time.time()
            result = video_service.preprocess_video(materials, clip_duration=2)
            process_time = time.time() - start_time
            
            if result and len(result) == len(materials):
                # Validate all outputs
                all_valid = True
                total_size = 0
                for r in result:
                    if not os.path.exists(r.url):
                        all_valid = False
                        break
                    size = os.path.getsize(r.url)
                    if size < 1000:
                        all_valid = False
                        break
                    total_size += size
                
                if all_valid:
                    success_count += 1
                    print(f"✅ PASSED ({process_time:.2f}s, {total_size} total bytes)")
                else:
                    print(f"❌ FAILED - Some outputs invalid")
                
                # Cleanup
                for r in result:
                    if os.path.exists(r.url):
                        os.remove(r.url)
            else:
                print(f"❌ FAILED - Invalid multi-clip result")
        
        except Exception as e:
            print(f"❌ MULTI-CLIP ERROR: {e}")
    
    # Test 3: Edge cases
    print(f"\n🔥 EDGE CASE TEST: Problematic scenarios")
    print("-" * 50)
    
    if test_images:
        # Very short duration
        try:
            material = MaterialInfo()
            material.url = test_images[0]
            material.provider = "local"
            material.duration = 0
            
            total_tests += 1
            print(f"Very short duration (0.5s)... ", end="", flush=True)
            
            result = video_service.preprocess_video([material], clip_duration=0.5)
            
            if result and len(result) == 1 and os.path.exists(result[0].url):
                file_size = os.path.getsize(result[0].url)
                if file_size > 500:  # Should still produce some content
                    success_count += 1
                    print(f"✅ PASSED ({file_size} bytes)")
                else:
                    print(f"❌ FAILED - Too small ({file_size} bytes)")
                os.remove(result[0].url)
            else:
                print(f"❌ FAILED - No valid output")
        
        except Exception as e:
            print(f"❌ EDGE CASE ERROR: {e}")
    
    # Final Report
    print("\n" + "="*60)
    print("BLACK SCREEN BUG VALIDATION REPORT")
    print("="*60)
    
    success_rate = (success_count / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Tests Run: {total_tests}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_tests - success_count}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("\n🎉 BLACK SCREEN BUG FIX: ✅ VALIDATED")
        print("Single clip processing is working correctly!")
        status = "PASSED"
    elif success_rate >= 70:
        print("\n⚠️  BLACK SCREEN BUG FIX: 🟡 MOSTLY WORKING") 
        print("Some issues remain, but major improvement detected")
        status = "PARTIAL"
    else:
        print("\n❌ BLACK SCREEN BUG FIX: ❌ FAILED")
        print("Critical issues still present")
        status = "FAILED"
    
    # System information
    print(f"\nSystem Performance:")
    print(f"  CPU Usage: {psutil.cpu_percent()}%")
    print(f"  Memory Usage: {psutil.virtual_memory().percent}%")
    print(f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    print("\n" + "="*60)
    
    return status == "PASSED"

if __name__ == "__main__":
    success = test_black_screen_bug_fix()
    sys.exit(0 if success else 1)
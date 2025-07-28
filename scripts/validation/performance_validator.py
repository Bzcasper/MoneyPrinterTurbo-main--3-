#!/usr/bin/env python3
"""
MoneyPrinterTurbo Enhanced - Performance Validation Suite
Stage 7 Completion: Comprehensive performance testing and validation
"""

import json
import os
import sys
import time
import subprocess
import threading
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

class PerformanceValidator:
    """Comprehensive performance validation for MoneyPrinterTurbo Enhanced"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_suite": "MoneyPrinterTurbo Enhanced Performance Validation",
            "version": "2.0.0",
            "tests": {}
        }
    
    def test_import_performance(self):
        """Test import performance of critical modules"""
        print("📦 Testing import performance...")
        
        imports_to_test = [
            ("app.config.config", "Config System"),
            ("app.models.schema", "Schema Models"),
            ("app.services.hive_memory", "Hive Memory System"),
            ("app.services.gpu_manager", "GPU Manager"),
            ("app.services.video_pipeline", "Video Pipeline")
        ]
        
        import_results = {}
        
        for module_name, display_name in imports_to_test:
            try:
                start_time = time.time()
                __import__(module_name)
                import_time = time.time() - start_time
                import_results[display_name] = {
                    "success": True,
                    "import_time": import_time,
                    "status": "✅ OK" if import_time < 1.0 else "⚠️ SLOW"
                }
                print(f"  {display_name}: {import_time:.3f}s")
            except Exception as e:
                import_results[display_name] = {
                    "success": False,
                    "error": str(e),
                    "status": "❌ FAILED"
                }
                print(f"  {display_name}: ❌ {e}")
        
        self.results["tests"]["import_performance"] = import_results
        return import_results
    
    def test_gpu_initialization(self):
        """Test GPU initialization and detection performance"""
        print("🎮 Testing GPU initialization...")
        
        try:
            from app.services.gpu_manager import get_gpu_manager
            
            start_time = time.time()
            gpu_manager = get_gpu_manager()
            init_time = time.time() - start_time
            
            start_time = time.time()
            gpus = gpu_manager.get_available_gpus()
            detection_time = time.time() - start_time
            
            gpu_results = {
                "initialization_time": init_time,
                "detection_time": detection_time,
                "gpus_found": len(gpus),
                "gpu_details": []
            }
            
            for gpu in gpus:
                gpu_results["gpu_details"].append({
                    "name": gpu.name,
                    "vendor": gpu.vendor.value,
                    "memory_total": gpu.memory_total,
                    "memory_free": gpu.memory_free
                })
            
            print(f"  Initialization: {init_time:.3f}s")
            print(f"  Detection: {detection_time:.3f}s")
            print(f"  GPUs found: {len(gpus)}")
            
            self.results["tests"]["gpu_performance"] = gpu_results
            return gpu_results
            
        except Exception as e:
            gpu_results = {"error": str(e), "success": False}
            print(f"  GPU test failed: {e}")
            self.results["tests"]["gpu_performance"] = gpu_results
            return gpu_results
    
    def test_hive_memory_performance(self):
        """Test Hive Memory system performance"""
        print("🧠 Testing Hive Memory performance...")
        
        try:
            from app.services.hive_memory import HiveMemoryManager
            
            # Initialize
            start_time = time.time()
            hive = HiveMemoryManager()
            init_time = time.time() - start_time
            
            # Test write operations
            test_data = {
                "test_type": "performance_validation",
                "timestamp": time.time(),
                "data": list(range(100))  # Some test data
            }
            
            write_times = []
            for i in range(5):
                start_time = time.time()
                hive.store_swarm_memory(f"perf_test_{i}", test_data, ttl=300)
                write_times.append(time.time() - start_time)
            
            # Test read operations
            read_times = []
            for i in range(5):
                start_time = time.time()
                result = hive.retrieve_swarm_memory(f"perf_test_{i}")
                read_times.append(time.time() - start_time)
            
            avg_write = sum(write_times) / len(write_times)
            avg_read = sum(read_times) / len(read_times)
            
            memory_results = {
                "initialization_time": init_time,
                "average_write_time": avg_write,
                "average_read_time": avg_read,
                "operations_tested": 10,
                "success": True
            }
            
            print(f"  Initialization: {init_time:.3f}s")
            print(f"  Average write: {avg_write:.3f}s")
            print(f"  Average read: {avg_read:.3f}s")
            
            self.results["tests"]["hive_memory_performance"] = memory_results
            return memory_results
            
        except Exception as e:
            memory_results = {"error": str(e), "success": False}
            print(f"  Hive Memory test failed: {e}")
            self.results["tests"]["hive_memory_performance"] = memory_results
            return memory_results
    
    def test_integration_cli(self):
        """Test enhanced integration CLI performance"""
        print("🔧 Testing Integration CLI...")
        
        try:
            if not Path("enhanced_integration.py").exists():
                raise FileNotFoundError("enhanced_integration.py not found")
            
            # Test CLI responsiveness
            start_time = time.time()
            result = subprocess.run([
                "python3", "enhanced_integration.py", "report"
            ], capture_output=True, text=True, timeout=30)
            cli_time = time.time() - start_time
            
            cli_results = {
                "response_time": cli_time,
                "exit_code": result.returncode,
                "success": result.returncode == 0,
                "output_length": len(result.stdout) if result.stdout else 0
            }
            
            if result.returncode == 0:
                print(f"  CLI response time: {cli_time:.3f}s")
                print(f"  Output generated: {len(result.stdout)} characters")
            else:
                print(f"  CLI test failed with exit code: {result.returncode}")
            
            self.results["tests"]["integration_cli"] = cli_results
            return cli_results
            
        except Exception as e:
            cli_results = {"error": str(e), "success": False}
            print(f"  Integration CLI test failed: {e}")
            self.results["tests"]["integration_cli"] = cli_results
            return cli_results
    
    def test_file_system_performance(self):
        """Test file system performance for video processing"""
        print("💽 Testing file system performance...")
        
        try:
            # Test write performance
            test_file = Path("perf_test_file.tmp")
            test_data = b"0" * (1024 * 1024)  # 1MB of data
            
            start_time = time.time()
            with open(test_file, "wb") as f:
                for _ in range(10):  # Write 10MB total
                    f.write(test_data)
            write_time = time.time() - start_time
            
            # Test read performance
            start_time = time.time()
            with open(test_file, "rb") as f:
                data = f.read()
            read_time = time.time() - start_time
            
            # Cleanup
            test_file.unlink()
            
            write_speed = 10 / write_time  # MB/s
            read_speed = 10 / read_time    # MB/s
            
            fs_results = {
                "write_speed_mbps": write_speed,
                "read_speed_mbps": read_speed,
                "write_time": write_time,
                "read_time": read_time,
                "test_size_mb": 10
            }
            
            print(f"  Write speed: {write_speed:.1f} MB/s")
            print(f"  Read speed: {read_speed:.1f} MB/s")
            
            self.results["tests"]["file_system_performance"] = fs_results
            return fs_results
            
        except Exception as e:
            fs_results = {"error": str(e), "success": False}
            print(f"  File system test failed: {e}")
            self.results["tests"]["file_system_performance"] = fs_results
            return fs_results
    
    def run_comprehensive_validation(self):
        """Run all performance validation tests"""
        print("🚀 MoneyPrinterTurbo Enhanced - Performance Validation Suite")
        print("=" * 60)
        
        # Run all tests
        self.test_import_performance()
        print()
        self.test_gpu_initialization()
        print()
        self.test_hive_memory_performance()
        print()
        self.test_integration_cli()
        print()
        self.test_file_system_performance()
        
        # Calculate overall performance score
        successful_tests = sum(1 for test in self.results["tests"].values() 
                             if isinstance(test, dict) and test.get("success", True))
        total_tests = len(self.results["tests"])
        performance_score = (successful_tests / total_tests) * 100
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "performance_score": performance_score,
            "overall_status": "EXCELLENT" if performance_score >= 90 else 
                            "GOOD" if performance_score >= 70 else
                            "NEEDS_ATTENTION"
        }
        
        # Save results
        report_file = f"performance_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print("\n📊 PERFORMANCE VALIDATION SUMMARY")
        print("=" * 40)
        print(f"✅ Successful tests: {successful_tests}/{total_tests}")
        print(f"📈 Performance score: {performance_score:.1f}%")
        print(f"🎯 Overall status: {self.results['summary']['overall_status']}")
        print(f"💾 Report saved: {report_file}")
        
        # Update TODO
        try:
            subprocess.run([
                "python3", "claude_cli.py", "todo-update",
                "--task", "Performance validation",
                "--status", "completed", 
                "--verify", f"performance score {performance_score:.1f}%, {successful_tests}/{total_tests} tests passed"
            ], check=True)
            print("📝 TODO.md updated with validation results")
        except:
            pass
        
        return self.results

def main():
    """Main entry point"""
    validator = PerformanceValidator()
    results = validator.run_comprehensive_validation()
    
    # Return appropriate exit code
    if results["summary"]["performance_score"] >= 70:
        print("\n🎉 Performance validation PASSED!")
        sys.exit(0)
    else:
        print("\n⚠️ Performance validation needs attention")
        sys.exit(1)

if __name__ == "__main__":
    main()

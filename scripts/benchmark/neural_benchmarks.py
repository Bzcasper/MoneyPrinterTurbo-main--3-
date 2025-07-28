"""
Neural Model Benchmarking System for MoneyPrinterTurbo
======================================================

Comprehensive benchmarking system for neural video enhancement models including:
- Inference speed and memory usage benchmarks
- Quality metrics evaluation (PSNR, SSIM, LPIPS)
- Model comparison and A/B testing results
- Hardware performance profiling
- Batch processing benchmarks

Author: ML Model Developer
Version: 1.0.0
"""

import os
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import statistics
import threading

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

from loguru import logger
from app.services.neural_training.model_integration import get_neural_processor
from app.services.neural_training.training_infrastructure import ModelType, get_model_versioning
from app.services.gpu_manager import get_gpu_manager


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    test_dataset_path: str
    output_dir: str = "benchmark_results"
    num_test_samples: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    input_resolutions: List[Tuple[int, int]] = field(default_factory=lambda: [(256, 256), (512, 512), (1024, 1024)])
    warmup_iterations: int = 10
    benchmark_iterations: int = 50
    enable_quality_metrics: bool = True
    enable_memory_profiling: bool = True
    enable_concurrent_testing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_dataset_path": self.test_dataset_path,
            "output_dir": self.output_dir,
            "num_test_samples": self.num_test_samples,
            "batch_sizes": self.batch_sizes,
            "input_resolutions": self.input_resolutions,
            "warmup_iterations": self.warmup_iterations,
            "benchmark_iterations": self.benchmark_iterations,
            "enable_quality_metrics": self.enable_quality_metrics,
            "enable_memory_profiling": self.enable_memory_profiling,
            "enable_concurrent_testing": self.enable_concurrent_testing
        }


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    model_id: str
    model_type: str
    batch_size: int
    input_resolution: Tuple[int, int]
    
    # Performance metrics
    avg_inference_time: float
    median_inference_time: float
    p95_inference_time: float
    throughput_fps: float
    
    # Memory metrics
    peak_memory_usage: float
    avg_memory_usage: float
    
    # Quality metrics (if available)
    avg_psnr: Optional[float] = None
    avg_ssim: Optional[float] = None
    avg_lpips: Optional[float] = None
    
    # Additional metadata
    device: str = "cpu"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "batch_size": self.batch_size,
            "input_resolution": self.input_resolution,
            "avg_inference_time": self.avg_inference_time,
            "median_inference_time": self.median_inference_time,
            "p95_inference_time": self.p95_inference_time,
            "throughput_fps": self.throughput_fps,
            "peak_memory_usage": self.peak_memory_usage,
            "avg_memory_usage": self.avg_memory_usage,
            "avg_psnr": self.avg_psnr,
            "avg_ssim": self.avg_ssim,
            "avg_lpips": self.avg_lpips,
            "device": self.device,
            "timestamp": self.timestamp.isoformat()
        }


class QualityMetrics:
    """Quality assessment metrics for video enhancement"""
    
    def __init__(self):
        self.lpips_model = None
        if LPIPS_AVAILABLE:
            try:
                self.lpips_model = lpips.LPIPS(net='alex')
                logger.info("LPIPS model loaded for perceptual quality assessment")
            except Exception as e:
                logger.warning(f"Failed to load LPIPS model: {e}")
                self.lpips_model = None
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        try:
            # Ensure images are in same range [0, 1]
            if img1.max() > 1.0:
                img1 = img1.astype(np.float32) / 255.0
            if img2.max() > 1.0:
                img2 = img2.astype(np.float32) / 255.0
            
            return psnr(img1, img2, data_range=1.0)
        except Exception as e:
            logger.warning(f"PSNR calculation failed: {e}")
            return 0.0
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        try:
            # Ensure images are in same range [0, 1]
            if img1.max() > 1.0:
                img1 = img1.astype(np.float32) / 255.0
            if img2.max() > 1.0:
                img2 = img2.astype(np.float32) / 255.0
            
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            else:
                img1_gray = img1
                img2_gray = img2
            
            return ssim(img1_gray, img2_gray, data_range=1.0)
        except Exception as e:
            logger.warning(f"SSIM calculation failed: {e}")
            return 0.0
    
    def calculate_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Learned Perceptual Image Patch Similarity"""
        if self.lpips_model is None:
            return 0.0
        
        try:
            # Convert to tensors
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            # Ensure images are in correct format
            if img1.max() <= 1.0:
                img1 = (img1 * 255).astype(np.uint8)
            if img2.max() <= 1.0:
                img2 = (img2 * 255).astype(np.uint8)
            
            tensor1 = transform(img1).unsqueeze(0)
            tensor2 = transform(img2).unsqueeze(0)
            
            with torch.no_grad():
                distance = self.lpips_model(tensor1, tensor2)
            
            return distance.item()
        except Exception as e:
            logger.warning(f"LPIPS calculation failed: {e}")
            return 0.0


class NeuralModelBenchmark:
    """Comprehensive benchmarking system for neural models"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.neural_processor = get_neural_processor()
        self.gpu_manager = get_gpu_manager()
        self.quality_metrics = QualityMetrics()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test data
        self.test_samples = self._load_test_samples()
        
        logger.info(f"Neural model benchmark initialized with {len(self.test_samples)} test samples")
    
    def _load_test_samples(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Load test dataset for benchmarking"""
        test_samples = []
        dataset_path = Path(self.config.test_dataset_path)
        
        if not dataset_path.exists():
            logger.warning(f"Test dataset not found: {dataset_path}")
            # Generate synthetic test data
            return self._generate_synthetic_data()
        
        try:
            # Load image pairs from dataset
            input_dir = dataset_path / "input"
            target_dir = dataset_path / "target"
            
            if input_dir.exists() and target_dir.exists():
                input_images = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
                
                for i, input_path in enumerate(input_images[:self.config.num_test_samples]):
                    target_path = target_dir / input_path.name
                    
                    if target_path.exists():
                        # Load images
                        input_img = np.array(Image.open(input_path).convert('RGB'))
                        target_img = np.array(Image.open(target_path).convert('RGB'))
                        
                        test_samples.append((input_img, target_img))
            
            logger.info(f"Loaded {len(test_samples)} test image pairs")
            
        except Exception as e:
            logger.error(f"Failed to load test dataset: {e}")
            test_samples = self._generate_synthetic_data()
        
        return test_samples
    
    def _generate_synthetic_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate synthetic test data"""
        logger.info("Generating synthetic test data")
        
        test_samples = []
        
        for i in range(min(self.config.num_test_samples, 50)):
            # Generate random image
            img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            
            # Create degraded version as input
            degraded = cv2.GaussianBlur(img, (5, 5), 1.0)
            degraded = cv2.resize(degraded, (128, 128))
            degraded = cv2.resize(degraded, (256, 256))
            
            test_samples.append((degraded, img))
        
        return test_samples
    
    async def benchmark_model(
        self,
        model_id: str,
        model_type: str,
        batch_sizes: List[int] = None,
        resolutions: List[Tuple[int, int]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark a single model across different configurations"""
        
        if batch_sizes is None:
            batch_sizes = self.config.batch_sizes
        
        if resolutions is None:
            resolutions = self.config.input_resolutions
        
        logger.info(f"Benchmarking model: {model_id} ({model_type})")
        
        results = []
        
        for batch_size in batch_sizes:
            for resolution in resolutions:
                logger.info(f"Testing batch_size={batch_size}, resolution={resolution}")
                
                try:
                    result = await self._benchmark_configuration(
                        model_id, model_type, batch_size, resolution
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {model_id} (batch={batch_size}, res={resolution}): {e}")
                    continue
        
        # Save results
        self._save_benchmark_results(model_id, results)
        
        return results
    
    async def _benchmark_configuration(
        self,
        model_id: str,
        model_type: str,
        batch_size: int,
        resolution: Tuple[int, int]
    ) -> BenchmarkResult:
        """Benchmark a specific model configuration"""
        
        # Prepare test data
        test_data = self._prepare_test_data(batch_size, resolution)
        
        # Device info
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Warmup
        logger.debug(f"Warming up model: {self.config.warmup_iterations} iterations")
        for _ in range(self.config.warmup_iterations):
            try:
                await self.neural_processor.model_server.inference(model_id, test_data[0])
            except Exception:
                pass  # Ignore warmup errors
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Benchmark inference time
        inference_times = []
        memory_usage = []
        quality_scores = {'psnr': [], 'ssim': [], 'lpips': []}
        
        for i in range(self.config.benchmark_iterations):
            # Memory before inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / 1024**2
            else:
                memory_before = 0
            
            # Run inference
            start_time = time.time()
            
            try:
                result = await self.neural_processor.model_server.inference(model_id, test_data[i % len(test_data)])
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Memory after inference
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated() / 1024**2
                    memory_usage.append(memory_after - memory_before)
                
                # Calculate quality metrics if enabled
                if self.config.enable_quality_metrics and result['success']:
                    self._calculate_quality_metrics(
                        result['output'], 
                        test_data[i % len(test_data)], 
                        quality_scores
                    )
                
            except Exception as e:
                logger.warning(f"Inference failed in iteration {i}: {e}")
                continue
        
        # Calculate statistics
        if not inference_times:
            raise RuntimeError("No successful inference runs")
        
        avg_inference_time = statistics.mean(inference_times)
        median_inference_time = statistics.median(inference_times)
        p95_inference_time = np.percentile(inference_times, 95)
        throughput_fps = batch_size / avg_inference_time
        
        peak_memory = max(memory_usage) if memory_usage else 0
        avg_memory = statistics.mean(memory_usage) if memory_usage else 0
        
        # Quality metrics
        avg_psnr = statistics.mean(quality_scores['psnr']) if quality_scores['psnr'] else None
        avg_ssim = statistics.mean(quality_scores['ssim']) if quality_scores['ssim'] else None
        avg_lpips = statistics.mean(quality_scores['lpips']) if quality_scores['lpips'] else None
        
        result = BenchmarkResult(
            model_id=model_id,
            model_type=model_type,
            batch_size=batch_size,
            input_resolution=resolution,
            avg_inference_time=avg_inference_time,
            median_inference_time=median_inference_time,
            p95_inference_time=p95_inference_time,
            throughput_fps=throughput_fps,
            peak_memory_usage=peak_memory,
            avg_memory_usage=avg_memory,
            avg_psnr=avg_psnr,
            avg_ssim=avg_ssim,
            avg_lpips=avg_lpips,
            device=device
        )
        
        logger.info(
            f"Benchmark result: {avg_inference_time:.3f}s avg, "
            f"{throughput_fps:.1f} FPS, {peak_memory:.1f}MB peak memory"
        )
        
        return result
    
    def _prepare_test_data(self, batch_size: int, resolution: Tuple[int, int]) -> List[np.ndarray]:
        """Prepare test data for benchmarking"""
        test_data = []
        
        for i in range(min(len(self.test_samples), 10)):  # Limit to 10 samples for benchmarking
            input_img, _ = self.test_samples[i]
            
            # Resize to target resolution
            resized_img = cv2.resize(input_img, resolution)
            
            # Create batch
            if batch_size == 1:
                test_data.append(resized_img)
            else:
                batch = np.stack([resized_img] * batch_size)
                test_data.append(batch)
        
        return test_data
    
    def _calculate_quality_metrics(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        quality_scores: Dict[str, List[float]]
    ):
        """Calculate quality metrics for prediction vs target"""
        try:
            # Handle batch dimension
            if len(prediction.shape) == 4:  # Batch
                pred = prediction[0]  # Take first sample
            else:
                pred = prediction
            
            if len(target.shape) == 4:  # Batch
                tgt = target[0]  # Take first sample
            else:
                tgt = target
            
            # Calculate PSNR
            psnr_score = self.quality_metrics.calculate_psnr(pred, tgt)
            quality_scores['psnr'].append(psnr_score)
            
            # Calculate SSIM
            ssim_score = self.quality_metrics.calculate_ssim(pred, tgt)
            quality_scores['ssim'].append(ssim_score)
            
            # Calculate LPIPS
            lpips_score = self.quality_metrics.calculate_lpips(pred, tgt)
            quality_scores['lpips'].append(lpips_score)
            
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
    
    def _save_benchmark_results(self, model_id: str, results: List[BenchmarkResult]):
        """Save benchmark results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"benchmark_{model_id}_{timestamp}.json"
        
        results_data = {
            'model_id': model_id,
            'timestamp': timestamp,
            'config': self.config.to_dict(),
            'results': [result.to_dict() for result in results]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved: {results_file}")
    
    async def benchmark_all_models(self) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark all loaded models"""
        logger.info("Benchmarking all loaded models")
        
        loaded_models = self.neural_processor.model_server.list_loaded_models()
        all_results = {}
        
        for model_info in loaded_models:
            model_id = model_info['model_id']
            model_type = model_info['model_type']
            
            try:
                results = await self.benchmark_model(model_id, model_type)
                all_results[model_id] = results
                
            except Exception as e:
                logger.error(f"Failed to benchmark model {model_id}: {e}")
                continue
        
        # Generate comparison report
        self._generate_comparison_report(all_results)
        
        return all_results
    
    def _generate_comparison_report(self, all_results: Dict[str, List[BenchmarkResult]]):
        """Generate model comparison report"""
        logger.info("Generating model comparison report")
        
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_results': {}
        }
        
        for model_id, results in all_results.items():
            if not results:
                continue
            
            # Calculate summary statistics
            avg_times = [r.avg_inference_time for r in results]
            throughputs = [r.throughput_fps for r in results]
            memory_usage = [r.peak_memory_usage for r in results]
            
            comparison_data['summary'][model_id] = {
                'avg_inference_time': statistics.mean(avg_times),
                'max_throughput': max(throughputs),
                'avg_memory_usage': statistics.mean(memory_usage),
                'num_configs_tested': len(results)
            }
            
            comparison_data['detailed_results'][model_id] = [r.to_dict() for r in results]
        
        # Save comparison report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"model_comparison_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.success(f"Model comparison report saved: {report_file}")
        
        # Print summary to console
        self._print_comparison_summary(comparison_data['summary'])
    
    def _print_comparison_summary(self, summary: Dict[str, Dict[str, float]]):
        """Print comparison summary to console"""
        logger.info("\n" + "="*80)
        logger.info("MODEL PERFORMANCE COMPARISON SUMMARY")
        logger.info("="*80)
        
        for model_id, stats in summary.items():
            logger.info(f"\n{model_id}:")
            logger.info(f"  Average Inference Time: {stats['avg_inference_time']:.3f}s")
            logger.info(f"  Max Throughput:         {stats['max_throughput']:.1f} FPS")
            logger.info(f"  Average Memory Usage:   {stats['avg_memory_usage']:.1f} MB")
            logger.info(f"  Configurations Tested:  {stats['num_configs_tested']}")
        
        logger.info("="*80)


async def run_neural_benchmarks(
    test_dataset_path: str,
    output_dir: str = "benchmark_results",
    num_samples: int = 100
) -> Dict[str, Any]:
    """Run comprehensive neural model benchmarks"""
    
    config = BenchmarkConfig(
        test_dataset_path=test_dataset_path,
        output_dir=output_dir,
        num_test_samples=num_samples
    )
    
    benchmark = NeuralModelBenchmark(config)
    results = await benchmark.benchmark_all_models()
    
    return {
        'benchmark_completed': True,
        'timestamp': datetime.now().isoformat(),
        'total_models_tested': len(results),
        'output_directory': output_dir,
        'results_summary': {
            model_id: len(model_results) 
            for model_id, model_results in results.items()
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Model Benchmarking")
    parser.add_argument("--dataset", required=True, help="Path to test dataset")
    parser.add_argument("--output", default="benchmark_results", help="Output directory")
    parser.add_argument("--samples", type=int, default=100, help="Number of test samples")
    
    args = parser.parse_args()
    
    async def main():
        await run_neural_benchmarks(args.dataset, args.output, args.samples)
    
    asyncio.run(main())
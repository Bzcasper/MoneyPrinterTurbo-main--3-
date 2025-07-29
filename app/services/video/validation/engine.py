"""
Validation Engine - Comprehensive input and file validation

This module provides comprehensive validation for video processing inputs,
including file validation, parameter validation, and codec compatibility checks.

Author: MoneyPrinterTurbo Enhanced System
Version: 1.0.0
"""

import os
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from datetime import datetime

from loguru import logger

from app.security.input_validator import InputValidator


class ValidationStatus(Enum):
    """Enumeration for validation status levels"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of validation operation"""
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    validator: str = "unknown"
    
    # Legacy compatibility fields
    is_valid: bool = field(init=False)
    errors: List[str] = field(default_factory=list, init=False)
    warnings: List[str] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        # Maintain backward compatibility
        self.is_valid = self.status == ValidationStatus.PASSED
        if self.details:
            self.errors = self.details.get('errors', [])
            self.warnings = self.details.get('warnings', [])


@dataclass
class FileValidationResult:
    """Result of file validation"""
    file_path: str
    is_valid: bool
    file_size: int
    file_format: str
    resolution: Optional[tuple] = None
    duration: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Legacy compatibility fields
    error: Optional[str] = None
    warning: Optional[str] = None
    file_info: Optional[Dict] = None


class ValidationEngine:
    """
    Comprehensive validation engine for video processing
    
    Handles input validation, file validation, and codec compatibility
    with security-first approach and detailed error reporting.
    """
    
    def __init__(self):
        """Initialize validation engine"""
        self.input_validator = InputValidator()
        self.file_validator = FileValidator()
        self.parameter_validator = ParameterValidator()
        self.codec_validator = CodecValidator()
        
        # Configuration limits
        self.max_file_size = 5 * 1024 * 1024 * 1024  # 5GB
        self.min_width = 64
        self.min_height = 64
        self.max_width = 7680  # 8K
        self.max_height = 4320  # 8K
        self.max_aspect_ratio = 10.0
        self.min_aspect_ratio = 0.1
        
        logger.info("ValidationEngine initialized successfully")
    
    def validate_inputs(self, params) -> ValidationResult:
        """
        Validate complete video processing inputs
        
        Args:
            params: VideoParams object containing processing parameters
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        
        try:
            # Validate basic parameters
            param_result = self.parameter_validator.validate(params)
            errors.extend(param_result.errors)
            warnings.extend(param_result.warnings)
            
            # Validate each input file
            for clip_path in params.clips:
                file_result = self.file_validator.validate_video_file(clip_path)
                if not file_result.is_valid:
                    errors.append(f"Invalid file: {clip_path} - {file_result.error}")
                elif file_result.warning:
                    warnings.append(f"File warning: {clip_path} - {file_result.warning}")
            
            # Validate codec compatibility
            codec_result = self.codec_validator.validate_compatibility(
                input_formats=self._extract_formats(params.clips),
                output_format=getattr(params, 'output_format', 'mp4')
            )
            
            if not codec_result.is_valid:
                errors.extend(codec_result.errors)
            warnings.extend(codec_result.warnings)
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )
    
    def validate_output(self, output_path: str) -> ValidationResult:
        """
        Validate processing output file
        
        Args:
            output_path: Path to output file
            
        Returns:
            ValidationResult with validation outcome
        """
        try:
            if not self.file_validator.file_exists(output_path):
                return ValidationResult(
                    is_valid=False,
                    errors=["Output file was not created"]
                )
            
            file_result = self.file_validator.validate_video_file(output_path)
            
            return ValidationResult(
                is_valid=file_result.is_valid,
                errors=[file_result.error] if file_result.error else [],
                warnings=[file_result.warning] if file_result.warning else []
            )
            
        except Exception as e:
            logger.error(f"Output validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Output validation error: {str(e)}"]
            )
    
    def is_healthy(self) -> bool:
        """Check if validation engine is healthy"""
        try:
            # Test basic functionality
            test_result = ValidationResult(is_valid=True)
            return test_result.is_valid
        except Exception:
            return False
    
    def _extract_formats(self, file_paths: List[str]) -> List[str]:
        """Extract file formats from file paths"""
        formats = []
        for path in file_paths:
            try:
                suffix = Path(path).suffix.lower().lstrip('.')
                if suffix:
                    formats.append(suffix)
            except Exception:
                continue
        return formats


class FileValidator:
    """Detailed file validation for video files"""
    
    def __init__(self):
        """Initialize file validator"""
        self.supported_formats = {
            'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v'
        }
        self.max_file_size = 5 * 1024 * 1024 * 1024  # 5GB
        self.min_width = 64
        self.min_height = 64
        self.max_width = 7680
        self.max_height = 4320
        self.max_aspect_ratio = 10.0
        self.min_aspect_ratio = 0.1
    
    def validate_video_file(self, file_path: str) -> FileValidationResult:
        """
        Validate individual video file
        
        Args:
            file_path: Path to video file
            
        Returns:
            FileValidationResult with validation outcome
        """
        # Basic file system checks
        if not self.file_exists(file_path):
            return FileValidationResult(
                is_valid=False,
                error="File does not exist"
            )
        
        file_size = self._get_file_size(file_path)
        if file_size == 0:
            return FileValidationResult(
                is_valid=False,
                error="File is empty (0 bytes)"
            )
        
        if file_size > self.max_file_size:
            return FileValidationResult(
                is_valid=False,
                error=f"File too large: {file_size} bytes > {self.max_file_size}"
            )
        
        # Format validation
        file_format = Path(file_path).suffix.lower().lstrip('.')
        if file_format not in self.supported_formats:
            return FileValidationResult(
                is_valid=False,
                error=f"Unsupported format: {file_format}"
            )
        
        # Video-specific validation using ffprobe
        video_info = self._probe_video_file(file_path)
        if not video_info:
            return FileValidationResult(
                is_valid=False,
                error="Failed to probe video file - may be corrupted"
            )
        
        # Validate dimensions
        width = video_info.get('width', 0)
        height = video_info.get('height', 0)
        
        if width < self.min_width or height < self.min_height:
            return FileValidationResult(
                is_valid=False,
                error=f"Dimensions too small: {width}x{height}"
            )
        
        if width > self.max_width or height > self.max_height:
            return FileValidationResult(
                is_valid=False,
                error=f"Dimensions too large: {width}x{height}"
            )
        
        # Validate aspect ratio
        if height > 0:
            aspect_ratio = width / height
            if aspect_ratio > self.max_aspect_ratio or aspect_ratio < self.min_aspect_ratio:
                return FileValidationResult(
                    is_valid=True,
                    warning=f"Extreme aspect ratio: {aspect_ratio:.2f}"
                )
        
        # Validate duration
        duration = video_info.get('duration', 0)
        if duration <= 0:
            return FileValidationResult(
                is_valid=False,
                error=f"Invalid duration: {duration}"
            )
        
        return FileValidationResult(
            is_valid=True,
            file_info=video_info
        )
    
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists"""
        try:
            return os.path.isfile(file_path)
        except Exception:
            return False
    
    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(file_path)
        except Exception:
            return 0
    
    def _probe_video_file(self, file_path: str) -> Optional[Dict]:
        """Probe video file for metadata using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return None
            
            import json
            data = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return None
            
            return {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'duration': float(video_stream.get('duration', 0)),
                'codec': video_stream.get('codec_name', 'unknown'),
                'fps': self._parse_fps(video_stream.get('r_frame_rate', '0/1'))
            }
            
        except Exception as e:
            logger.warning(f"Failed to probe video file {file_path}: {str(e)}")
            return None
    
    def _parse_fps(self, fps_string: str) -> float:
        """Parse frame rate string like '30/1' to float"""
        try:
            if '/' in fps_string:
                num, den = fps_string.split('/')
                return float(num) / float(den) if float(den) != 0 else 0.0
            return float(fps_string)
        except Exception:
            return 0.0


class ParameterValidator:
    """Validate processing parameters"""
    
    def validate(self, params) -> ValidationResult:
        """Validate video processing parameters"""
        errors = []
        warnings = []
        
        # Validate clips
        if not hasattr(params, 'clips') or not params.clips:
            errors.append("No input clips provided")
        elif len(params.clips) > 100:  # Reasonable limit
            errors.append("Too many clips (max 100)")
        
        # Validate dimensions
        if hasattr(params, 'dimensions'):
            dims = params.dimensions
            if not isinstance(dims, dict):
                errors.append("Dimensions must be a dictionary")
            else:
                width = dims.get('width', 0)
                height = dims.get('height', 0)
                if width <= 0 or height <= 0:
                    errors.append("Invalid dimensions")
        
        # Validate output path
        if hasattr(params, 'output_path'):
            if not params.output_path:
                errors.append("Output path is required")
            elif not isinstance(params.output_path, str):
                errors.append("Output path must be a string")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class CodecValidator:
    """Validate codec compatibility"""
    
    def __init__(self):
        """Initialize codec validator"""
        self.compatible_combinations = {
            'mp4': ['h264', 'h265', 'aac'],
            'avi': ['h264', 'xvid', 'mp3'],
            'mkv': ['h264', 'h265', 'vp9', 'aac', 'opus'],
            'webm': ['vp8', 'vp9', 'opus']
        }
    
    def validate_compatibility(
        self, input_formats: List[str], output_format: str
    ) -> ValidationResult:
        """Validate codec compatibility"""
        errors = []
        warnings = []
        
        if output_format not in self.compatible_combinations:
            warnings.append(f"Unknown output format: {output_format}")
        
        # Check for format consistency
        unique_formats = set(input_formats)
        if len(unique_formats) > 3:
            warnings.append("Many different input formats may cause compatibility issues")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
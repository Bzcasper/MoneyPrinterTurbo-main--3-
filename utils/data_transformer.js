/**
 * Data Transformation Utility
 * Provides functions for processing, validating, and transforming data structures
 * commonly used in video generation and content processing workflows
 */

/**
 * Transform and validate video generation parameters
 * @param {Object} rawParams - Raw input parameters
 * @param {Object} schema - Validation schema
 * @returns {Object} Transformed and validated parameters
 */
function transformVideoParams(rawParams, schema = {}) {
  if (!rawParams || typeof rawParams !== 'object') {
    throw new Error('Raw parameters must be a valid object');
  }

  const defaults = {
    duration: 30,
    resolution: '1920x1080',
    fps: 30,
    format: 'mp4',
    quality: 'high'
  };

  const transformed = { ...defaults };

  // Transform duration
  if (rawParams.duration !== undefined) {
    const duration = parseFloat(rawParams.duration);
    if (isNaN(duration) || duration <= 0 || duration > 3600) {
      throw new Error('Duration must be a positive number between 0 and 3600 seconds');
    }
    transformed.duration = duration;
  }

  // Transform resolution
  if (rawParams.resolution) {
    const resolutionPattern = /^(\d+)x(\d+)$/;
    if (!resolutionPattern.test(rawParams.resolution)) {
      throw new Error('Resolution must be in format "widthxheight" (e.g., "1920x1080")');
    }
    const [, width, height] = rawParams.resolution.match(resolutionPattern);
    if (parseInt(width) < 480 || parseInt(height) < 360) {
      throw new Error('Minimum resolution is 480x360');
    }
    transformed.resolution = rawParams.resolution;
  }

  // Transform FPS
  if (rawParams.fps !== undefined) {
    const fps = parseInt(rawParams.fps);
    if (isNaN(fps) || fps < 1 || fps > 120) {
      throw new Error('FPS must be a number between 1 and 120');
    }
    transformed.fps = fps;
  }

  // Transform format
  if (rawParams.format) {
    const validFormats = ['mp4', 'avi', 'mov', 'webm', 'mkv'];
    if (!validFormats.includes(rawParams.format.toLowerCase())) {
      throw new Error(`Format must be one of: ${validFormats.join(', ')}`);
    }
    transformed.format = rawParams.format.toLowerCase();
  }

  // Transform quality
  if (rawParams.quality) {
    const validQualities = ['low', 'medium', 'high', 'ultra'];
    if (!validQualities.includes(rawParams.quality.toLowerCase())) {
      throw new Error(`Quality must be one of: ${validQualities.join(', ')}`);
    }
    transformed.quality = rawParams.quality.toLowerCase();
  }

  return transformed;
}

/**
 * Process and normalize text content for video generation
 * @param {string|Array} content - Text content to process
 * @param {Object} options - Processing options
 * @returns {Object} Processed content with metadata
 */
function processTextContent(content, options = {}) {
  const {
    maxLength = 5000,
    minLength = 10,
    splitSentences = true,
    removeEmptyLines = true,
    normalizeWhitespace = true
  } = options;

  if (!content) {
    throw new Error('Content is required');
  }

  let processedText = Array.isArray(content) ? content.join('\n') : String(content);

  // Normalize whitespace
  if (normalizeWhitespace) {
    processedText = processedText.replace(/\s+/g, ' ').trim();
  }

  // Remove empty lines
  if (removeEmptyLines) {
    processedText = processedText.split('\n')
      .filter(line => line.trim().length > 0)
      .join('\n');
  }

  // Validate length
  if (processedText.length < minLength) {
    throw new Error(`Content must be at least ${minLength} characters long`);
  }

  if (processedText.length > maxLength) {
    throw new Error(`Content must not exceed ${maxLength} characters`);
  }

  const result = {
    text: processedText,
    wordCount: processedText.split(/\s+/).length,
    characterCount: processedText.length,
    estimatedReadingTime: Math.ceil(processedText.split(/\s+/).length / 200), // 200 WPM average
  };

  // Split into sentences if requested
  if (splitSentences) {
    result.sentences = processedText.split(/[.!?]+/)
      .map(s => s.trim())
      .filter(s => s.length > 0);
    result.sentenceCount = result.sentences.length;
  }

  return result;
}

/**
 * Validate and transform audio parameters
 * @param {Object} audioParams - Audio parameters to validate
 * @returns {Object} Validated audio parameters
 */
function transformAudioParams(audioParams = {}) {
  const defaults = {
    bitrate: 128,
    sampleRate: 44100,
    channels: 2,
    format: 'mp3',
    volume: 1.0
  };

  const transformed = { ...defaults, ...audioParams };

  // Validate bitrate
  const validBitrates = [64, 96, 128, 192, 256, 320];
  if (!validBitrates.includes(transformed.bitrate)) {
    throw new Error(`Bitrate must be one of: ${validBitrates.join(', ')} kbps`);
  }

  // Validate sample rate
  const validSampleRates = [22050, 44100, 48000, 96000];
  if (!validSampleRates.includes(transformed.sampleRate)) {
    throw new Error(`Sample rate must be one of: ${validSampleRates.join(', ')} Hz`);
  }

  // Validate channels
  if (![1, 2].includes(transformed.channels)) {
    throw new Error('Channels must be 1 (mono) or 2 (stereo)');
  }

  // Validate format
  const validAudioFormats = ['mp3', 'wav', 'aac', 'ogg'];
  if (!validAudioFormats.includes(transformed.format)) {
    throw new Error(`Audio format must be one of: ${validAudioFormats.join(', ')}`);
  }

  // Validate volume
  if (transformed.volume < 0 || transformed.volume > 2) {
    throw new Error('Volume must be between 0.0 and 2.0');
  }

  return transformed;
}

/**
 * Deep merge objects with type validation
 * @param {Object} target - Target object
 * @param {Object} source - Source object to merge
 * @param {Object} typeValidators - Optional type validators
 * @returns {Object} Merged object
 */
function deepMergeWithValidation(target, source, typeValidators = {}) {
  if (!target || typeof target !== 'object') {
    throw new Error('Target must be an object');
  }

  if (!source || typeof source !== 'object') {
    return { ...target };
  }

  const result = { ...target };

  for (const [key, value] of Object.entries(source)) {
    // Apply type validation if provided
    if (typeValidators[key]) {
      const validator = typeValidators[key];
      if (typeof validator === 'function') {
        if (!validator(value)) {
          throw new Error(`Validation failed for key "${key}"`);
        }
      } else if (typeof validator === 'string') {
        if (typeof value !== validator) {
          throw new Error(`Expected ${validator} for key "${key}", got ${typeof value}`);
        }
      }
    }

    if (value && typeof value === 'object' && !Array.isArray(value)) {
      result[key] = deepMergeWithValidation(result[key] || {}, value, typeValidators[key]);
    } else {
      result[key] = value;
    }
  }

  return result;
}

/**
 * Sanitize and validate file paths
 * @param {string} filePath - File path to sanitize
 * @param {Object} options - Sanitization options
 * @returns {string} Sanitized file path
 */
function sanitizeFilePath(filePath, options = {}) {
  const {
    allowedExtensions = [],
    maxLength = 255,
    replaceSpaces = true,
    removeSpecialChars = true
  } = options;

  if (!filePath || typeof filePath !== 'string') {
    throw new Error('File path must be a non-empty string');
  }

  let sanitized = filePath.trim();

  // Remove or replace invalid characters
  if (removeSpecialChars) {
    sanitized = sanitized.replace(/[<>:"|?*]/g, '');
  }

  // Replace spaces with underscores
  if (replaceSpaces) {
    sanitized = sanitized.replace(/\s+/g, '_');
  }

  // Validate length
  if (sanitized.length > maxLength) {
    throw new Error(`File path exceeds maximum length of ${maxLength} characters`);
  }

  // Validate file extension if specified
  if (allowedExtensions.length > 0) {
    const extension = sanitized.split('.').pop()?.toLowerCase();
    if (!extension || !allowedExtensions.includes(extension)) {
      throw new Error(`File extension must be one of: ${allowedExtensions.join(', ')}`);
    }
  }

  return sanitized;
}

module.exports = {
  transformVideoParams,
  processTextContent,
  transformAudioParams,
  deepMergeWithValidation,
  sanitizeFilePath
};
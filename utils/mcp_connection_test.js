/**
 * MCP Connection Test Utility
 * Provides functions to test and validate MCP server connections
 */

const https = require('https');
const http = require('http');

/**
 * Test MCP server connection with retry logic
 * @param {string} serverUrl - The MCP server URL to test
 * @param {Object} options - Configuration options
 * @param {number} options.timeout - Request timeout in milliseconds (default: 5000)
 * @param {number} options.retries - Number of retry attempts (default: 3)
 * @param {number} options.retryDelay - Delay between retries in milliseconds (default: 1000)
 * @returns {Promise<Object>} Connection test result
 */
async function testMcpConnection(serverUrl, options = {}) {
  const {
    timeout = 5000,
    retries = 3,
    retryDelay = 1000
  } = options;

  // Validate input
  if (!serverUrl || typeof serverUrl !== 'string') {
    throw new Error('Server URL is required and must be a string');
  }

  // Parse URL to determine protocol
  const url = new URL(serverUrl);
  const isHttps = url.protocol === 'https:';
  const requestLib = isHttps ? https : http;

  /**
   * Perform a single connection attempt
   */
  const attemptConnection = () => {
    return new Promise((resolve, reject) => {
      const requestOptions = {
        hostname: url.hostname,
        port: url.port || (isHttps ? 443 : 80),
        path: url.pathname + url.search,
        method: 'GET',
        timeout: timeout,
        headers: {
          'User-Agent': 'MCP-Connection-Tester/1.0',
          'Accept': 'application/json'
        }
      };

      const req = requestLib.request(requestOptions, (res) => {
        let data = '';
        
        res.on('data', (chunk) => {
          data += chunk;
        });

        res.on('end', () => {
          resolve({
            success: true,
            statusCode: res.statusCode,
            headers: res.headers,
            data: data,
            responseTime: Date.now() - startTime
          });
        });
      });

      req.on('error', (error) => {
        reject({
          success: false,
          error: error.message,
          code: error.code,
          responseTime: Date.now() - startTime
        });
      });

      req.on('timeout', () => {
        req.destroy();
        reject({
          success: false,
          error: 'Request timeout',
          code: 'TIMEOUT',
          responseTime: timeout
        });
      });

      const startTime = Date.now();
      req.end();
    });
  };

  // Retry logic
  let lastError;
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      console.log(`Attempting connection to ${serverUrl} (attempt ${attempt}/${retries})`);
      
      const result = await attemptConnection();
      
      return {
        ...result,
        attempt: attempt,
        serverUrl: serverUrl
      };
    } catch (error) {
      lastError = error;
      console.log(`Attempt ${attempt} failed:`, error.error || error.message);
      
      // Wait before retry (except for last attempt)
      if (attempt < retries) {
        console.log(`Waiting ${retryDelay}ms before retry...`);
        await new Promise(resolve => setTimeout(resolve, retryDelay));
      }
    }
  }

  // All attempts failed
  return {
    success: false,
    error: lastError.error || lastError.message,
    code: lastError.code || 'UNKNOWN_ERROR',
    attempts: retries,
    serverUrl: serverUrl
  };
}

/**
 * Test multiple MCP servers concurrently
 * @param {Array<string>} serverUrls - Array of server URLs to test
 * @param {Object} options - Configuration options (same as testMcpConnection)
 * @returns {Promise<Array<Object>>} Array of connection test results
 */
async function testMultipleMcpServers(serverUrls, options = {}) {
  if (!Array.isArray(serverUrls) || serverUrls.length === 0) {
    throw new Error('Server URLs must be a non-empty array');
  }

  console.log(`Testing ${serverUrls.length} MCP servers...`);
  
  const promises = serverUrls.map(url => testMcpConnection(url, options));
  const results = await Promise.allSettled(promises);
  
  return results.map((result, index) => ({
    serverUrl: serverUrls[index],
    ...result.value
  }));
}

/**
 * Validate MCP configuration format
 * @param {Object} config - MCP configuration object
 * @returns {Object} Validation result
 */
function validateMcpConfig(config) {
  const errors = [];
  const warnings = [];

  // Check basic structure
  if (!config || typeof config !== 'object') {
    errors.push('Configuration must be an object');
    return { valid: false, errors, warnings };
  }

  // Check servers array
  if (!config.servers || !Array.isArray(config.servers)) {
    errors.push('Configuration must have a "servers" array');
  } else {
    config.servers.forEach((server, index) => {
      if (!server.name) {
        errors.push(`Server at index ${index} missing "name" property`);
      }
      if (!server.url) {
        errors.push(`Server at index ${index} missing "url" property`);
      } else {
        try {
          new URL(server.url);
        } catch (e) {
          errors.push(`Server at index ${index} has invalid URL: ${server.url}`);
        }
      }
      if (!server.tools || !Array.isArray(server.tools)) {
        warnings.push(`Server at index ${index} has no tools defined`);
      }
    });
  }

  // Check for conflicting mcpServers section
  if (config.mcpServers) {
    warnings.push('Found "mcpServers" section - this may conflict with remote server configuration');
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
}

module.exports = {
  testMcpConnection,
  testMultipleMcpServers,
  validateMcpConfig
};

/**
 * Vercel-Compatible TypeScript MCP Client SDK
 * 
 * Provides TypeScript/JavaScript client for connecting to MCP servers
 * with full type safety, HTTP-based communication, and Vercel serverless compatibility.
 * Uses fetch API instead of WebSocket for maximum compatibility.
 */

// Base MCP protocol interfaces
export interface MCPMessage {
  jsonrpc: string;
  id?: string | number;
  timestamp?: string;
}

export interface MCPRequest extends MCPMessage {
  method: string;
  params?: Record<string, any>;
}

export interface MCPResponse extends MCPMessage {
  result?: any;
  error?: MCPError;
}

export interface MCPError {
  code: number;
  message: string;
  data?: Record<string, any>;
}

export interface MCPTool {
  name: string;
  description: string;
  input_schema: Record<string, any>;
  output_schema?: Record<string, any>;
  category?: string;
  version?: string;
}

// Explicit interface instead of Required<T> utility type
export interface MCPClientConfig {
  serverUrl: string;
  authType: 'api_key' | 'jwt' | 'hmac';
  apiKey?: string;
  jwtToken?: string;
  clientId?: string;
  clientSecret?: string;
  
  // Connection settings
  maxRetries: number;
  retryDelay: number;
  requestTimeout: number;
  
  // Cache settings
  enableCaching: boolean;
  cacheTtl: number;
  
  // Optional headers
  customHeaders?: Record<string, string>;
}

export enum ConnectionState {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  FAILED = 'failed'
}

// Browser-compatible timer handle type
type TimerHandle = ReturnType<typeof setTimeout>;

/**
 * Vercel-compatible MCP connection using HTTP/fetch instead of WebSocket
 */
class MCPHttpConnection {
  private config: MCPClientConfig;
  private state: ConnectionState = ConnectionState.DISCONNECTED;
  private lastActivity: number = Date.now();
  private authHeaders: Record<string, string> = {};

  constructor(config: MCPClientConfig) {
    this.config = config;
  }

  async connect(): Promise<boolean> {
    try {
      this.state = ConnectionState.CONNECTING;
      console.log(`Connecting to MCP server: ${this.config.serverUrl}`);

      // Authenticate and get session
      const authenticated = await this.authenticate();
      if (!authenticated) {
        this.state = ConnectionState.FAILED;
        return false;
      }

      this.state = ConnectionState.CONNECTED;
      console.log('Connected to MCP server');
      return true;
    } catch (error) {
      console.error(`Failed to connect to MCP server: ${error}`);
      this.state = ConnectionState.FAILED;
      return false;
    }
  }

  async disconnect(): Promise<void> {
    this.state = ConnectionState.DISCONNECTED;
    this.authHeaders = {};
    console.log('Disconnected from MCP server');
  }

  async sendRequest(request: MCPRequest): Promise<MCPResponse> {
    if (this.state !== ConnectionState.CONNECTED) {
      throw new Error('Not connected to MCP server');
    }

    const requestId = request.id || this.generateRequestId();
    request.id = requestId;
    request.timestamp = new Date().toISOString();

    const response = await this.makeHttpRequest(request);
    this.lastActivity = Date.now();
    return response;
  }

  private async makeHttpRequest(request: MCPRequest): Promise<MCPResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.requestTimeout);

    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        ...this.authHeaders,
        ...this.config.customHeaders
      };

      const response = await fetch(this.config.serverUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify(request),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateResponse(data);
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new Error(`Request timeout after ${this.config.requestTimeout}ms`);
      }
      throw error;
    }
  }

  private async authenticate(): Promise<boolean> {
    try {
      const authData: Record<string, any> = { type: this.config.authType };

      switch (this.config.authType) {
        case 'api_key':
          if (!this.config.apiKey) {
            throw new Error('API key required for api_key auth');
          }
          this.authHeaders['Authorization'] = `Bearer ${this.config.apiKey}`;
          break;

        case 'jwt':
          if (!this.config.jwtToken) {
            throw new Error('JWT token required for jwt auth');
          }
          this.authHeaders['Authorization'] = `Bearer ${this.config.jwtToken}`;
          break;

        case 'hmac':
          if (!this.config.clientId || !this.config.clientSecret) {
            throw new Error('Client ID and secret required for HMAC auth');
          }
          const timestamp = Math.floor(Date.now() / 1000).toString();
          const signature = await this.generateHmacSignature(timestamp);
          this.authHeaders['X-Client-ID'] = this.config.clientId;
          this.authHeaders['X-Timestamp'] = timestamp;
          this.authHeaders['X-Signature'] = signature;
          break;

        default:
          throw new Error(`Unsupported auth type: ${this.config.authType}`);
      }

      // Test authentication with a simple request
      const testRequest: MCPRequest = {
        jsonrpc: '2.0',
        method: 'auth/verify',
        params: authData
      };

      const response = await this.makeHttpRequest(testRequest);

      if (response.error) {
        console.error(`Authentication failed: ${response.error.message}`);
        return false;
      }

      console.log('Successfully authenticated with MCP server');
      return true;
    } catch (error) {
      console.error(`Authentication error: ${error}`);
      return false;
    }
  }

  private async generateHmacSignature(timestamp: string): Promise<string> {
    if (!this.config.clientSecret) {
      throw new Error('Client secret required for HMAC signature');
    }

    const message = `${this.config.clientId}${timestamp}`;
    const encoder = new TextEncoder();
    
    // Use browser-compatible crypto API
    const key = await crypto.subtle.importKey(
      'raw',
      encoder.encode(this.config.clientSecret),
      { name: 'HMAC', hash: 'SHA-256' },
      false,
      ['sign']
    );
    
    const signature = await crypto.subtle.sign('HMAC', key, encoder.encode(message));
    return Array.from(new Uint8Array(signature))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
  }

  private validateResponse(data: any): MCPResponse {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid response format');
    }
    if (data.jsonrpc !== '2.0') {
      throw new Error('Invalid JSON-RPC version');
    }
    return data as MCPResponse;
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  get isConnected(): boolean {
    return this.state === ConnectionState.CONNECTED;
  }

  get connectionState(): ConnectionState {
    return this.state;
  }
}

/**
 * Simple cache implementation for browsers
 */
class BrowserCache {
  private cache = new Map<string, { result: any; timestamp: number }>();
  private maxSize = 1000;

  set(key: string, result: any, ttl: number): void {
    // Clean old entries if cache is getting too large
    if (this.cache.size >= this.maxSize) {
      this.cleanup();
    }

    this.cache.set(key, {
      result,
      timestamp: Date.now()
    });
  }

  get(key: string, ttl: number): any | null {
    const cached = this.cache.get(key);
    if (!cached) {
      return null;
    }

    if (Date.now() - cached.timestamp > ttl) {
      this.cache.delete(key);
      return null;
    }

    return cached.result;
  }

  private cleanup(): void {
    // Remove oldest 10% of entries
    const entries = Array.from(this.cache.entries())
      .sort(([,a], [,b]) => a.timestamp - b.timestamp);
    
    const toRemove = Math.floor(entries.length * 0.1);
    for (let i = 0; i < toRemove; i++) {
      this.cache.delete(entries[i][0]);
    }
  }

  clear(): void {
    this.cache.clear();
  }
}

/**
 * Vercel-compatible MCP Client
 * Uses HTTP requests instead of WebSocket for maximum compatibility
 */
export class MCPClient {
  private config: MCPClientConfig;
  private connection: MCPHttpConnection;
  private cache: BrowserCache;
  private availableTools: MCPTool[] = [];

  constructor(config: Partial<MCPClientConfig> & { serverUrl: string }) {
    this.config = {
      authType: 'api_key',
      maxRetries: 3,
      retryDelay: 1000,
      requestTimeout: 30000,
      enableCaching: true,
      cacheTtl: 3600000, // 1 hour
      ...config
    } as MCPClientConfig;

    this.connection = new MCPHttpConnection(this.config);
    this.cache = new BrowserCache();
  }

  async connect(): Promise<void> {
    const connected = await this.connection.connect();
    if (!connected) {
      throw new Error('Failed to connect to MCP server');
    }

    // Load available tools
    await this.loadTools();
  }

  async disconnect(): Promise<void> {
    await this.connection.disconnect();
    this.cache.clear();
  }

  async callTool(toolName: string, parameters: Record<string, any>, useCache = true): Promise<any> {
    // Check cache first
    if (useCache && this.config.enableCaching) {
      const cacheKey = this.getCacheKey(toolName, parameters);
      const cachedResult = this.cache.get(cacheKey, this.config.cacheTtl);
      if (cachedResult) {
        return cachedResult;
      }
    }

    const request: MCPRequest = {
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: toolName,
        arguments: parameters
      }
    };

    const response = await this.sendWithRetry(request);

    if (response.error) {
      throw new Error(`Tool call failed: ${response.error.message}`);
    }

    const result = response.result;

    // Cache result
    if (useCache && this.config.enableCaching) {
      const cacheKey = this.getCacheKey(toolName, parameters);
      this.cache.set(cacheKey, result, this.config.cacheTtl);
    }

    return result;
  }

  async listTools(): Promise<MCPTool[]> {
    if (this.availableTools.length > 0) {
      return this.availableTools;
    }

    await this.loadTools();
    return this.availableTools;
  }

  async getServiceStatus(): Promise<any> {
    const request: MCPRequest = {
      jsonrpc: '2.0',
      method: 'service/status'
    };

    const response = await this.sendWithRetry(request);

    if (response.error) {
      throw new Error(`Status check failed: ${response.error.message}`);
    }

    return response.result;
  }

  private async loadTools(): Promise<void> {
    const request: MCPRequest = {
      jsonrpc: '2.0',
      method: 'tools/list'
    };

    const response = await this.sendWithRetry(request);

    if (response.error) {
      console.error(`Failed to load tools: ${response.error.message}`);
      return;
    }

    this.availableTools = response.result as MCPTool[];
    console.log(`Loaded ${this.availableTools.length} MCP tools`);
  }

  private async sendWithRetry(request: MCPRequest): Promise<MCPResponse> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      try {
        if (attempt > 0) {
          await this.sleep(this.config.retryDelay * attempt);
        }

        return await this.connection.sendRequest(request);
      } catch (error) {
        lastError = error as Error;
        console.warn(`Request attempt ${attempt + 1} failed: ${error}`);

        // Try to reconnect if connection failed
        if (!this.connection.isConnected) {
          await this.connection.connect();
        }
      }
    }

    throw lastError;
  }

  private getCacheKey(toolName: string, parameters: Record<string, any>): string {
    const keyData = `${toolName}:${JSON.stringify(parameters, Object.keys(parameters).sort())}`;
    // Simple hash function for browser compatibility
    let hash = 0;
    for (let i = 0; i < keyData.length; i++) {
      const char = keyData.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(36);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  get isConnected(): boolean {
    return this.connection.isConnected;
  }

  get connectionState(): ConnectionState {
    return this.connection.connectionState;
  }
}

// Convenience factory functions
export async function createMCPClient(
  serverUrl: string, 
  apiKey: string, 
  options: Partial<MCPClientConfig> = {}
): Promise<MCPClient> {
  const config = {
    serverUrl,
    authType: 'api_key' as const,
    apiKey,
    ...options
  };

  const client = new MCPClient(config);
  await client.connect();
  return client;
}

export async function createMCPClientJWT(
  serverUrl: string, 
  jwtToken: string, 
  options: Partial<MCPClientConfig> = {}
): Promise<MCPClient> {
  const config = {
    serverUrl,
    authType: 'jwt' as const,
    jwtToken,
    ...options
  };

  const client = new MCPClient(config);
  await client.connect();
  return client;
}

export async function createMCPClientHMAC(
  serverUrl: string, 
  clientId: string, 
  clientSecret: string, 
  options: Partial<MCPClientConfig> = {}
): Promise<MCPClient> {
  const config = {
    serverUrl,
    authType: 'hmac' as const,
    clientId,
    clientSecret,
    ...options
  };

  const client = new MCPClient(config);
  await client.connect();
  return client;
}

// Default export for easier importing
export default MCPClient;
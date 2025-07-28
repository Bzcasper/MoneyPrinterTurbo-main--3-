/**
 * TypeScript MCP Client SDK
 * 
 * Provides TypeScript/JavaScript client for connecting to MCP servers
 * with full type safety, connection pooling, and retry logic.
 */

interface MCPMessage {
  jsonrpc: string;
  id?: string | number;
  timestamp?: string;
}

interface MCPRequest extends MCPMessage {
  method: string;
  params?: Record<string, any>;
}

interface MCPResponse extends MCPMessage {
  result?: any;
  error?: MCPError;
}

interface MCPError {
  code: number;
  message: string;
  data?: Record<string, any>;
}

interface MCPTool {
  name: string;
  description: string;
  input_schema: Record<string, any>;
  output_schema?: Record<string, any>;
  category?: string;
  version?: string;
}

interface MCPClientConfig {
  serverUrl: string;
  authType?: 'api_key' | 'jwt' | 'hmac';
  apiKey?: string;
  jwtToken?: string;
  clientId?: string;
  clientSecret?: string;
  
  // Connection settings
  maxRetries?: number;
  retryDelay?: number;
  connectionTimeout?: number;
  heartbeatInterval?: number;
  
  // Pool settings
  maxConnections?: number;
  idleTimeout?: number;
  
  // Cache settings
  enableCaching?: boolean;
  cacheTtl?: number;
}

enum ConnectionState {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  RECONNECTING = 'reconnecting',
  FAILED = 'failed'
}

class MCPConnection {
  private config: MCPClientConfig;
  private websocket: WebSocket | null = null;
  private connectionId: string | null = null;
  private state: ConnectionState = ConnectionState.DISCONNECTED;
  private lastActivity: number = Date.now();
  private pendingRequests: Map<string, {
    resolve: (value: MCPResponse) => void;
    reject: (reason: any) => void;
    timeout: NodeJS.Timeout;
  }> = new Map();
  private eventHandlers: Map<string, Array<(data: any) => void>> = new Map();
  private heartbeatTimer: NodeJS.Timeout | null = null;

  constructor(config: MCPClientConfig) {
    this.config = {
      maxRetries: 3,
      retryDelay: 1000,
      connectionTimeout: 10000,
      heartbeatInterval: 30000,
      maxConnections: 5,
      idleTimeout: 300000,
      enableCaching: true,
      cacheTtl: 3600000,
      ...config
    };
  }

  async connect(): Promise<boolean> {
    try {
      this.state = ConnectionState.CONNECTING;
      console.log(`Connecting to MCP server: ${this.config.serverUrl}`);

      // Create WebSocket connection
      this.websocket = new WebSocket(this.config.serverUrl);
      
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Connection timeout'));
        }, this.config.connectionTimeout);

        this.websocket!.onopen = () => {
          clearTimeout(timeout);
          this.setupMessageHandlers();
          this.waitForWelcome().then(welcome => {
            if (welcome) {
              this.connectionId = welcome.result?.connection_id;
              this.state = ConnectionState.CONNECTED;
              console.log(`Connected to MCP server with ID: ${this.connectionId}`);
              
              this.authenticate().then(authenticated => {
                if (authenticated) {
                  this.startHeartbeat();
                  resolve(true);
                } else {
                  this.disconnect();
                  resolve(false);
                }
              }).catch(reject);
            } else {
              this.disconnect();
              resolve(false);
            }
          }).catch(reject);
        };

        this.websocket!.onerror = (error) => {
          clearTimeout(timeout);
          this.state = ConnectionState.FAILED;
          reject(error);
        };
      });
    } catch (error) {
      console.error(`Failed to connect to MCP server: ${error}`);
      this.state = ConnectionState.FAILED;
      await this.disconnect();
      return false;
    }
  }

  async disconnect(): Promise<void> {
    this.state = ConnectionState.DISCONNECTED;

    // Clear heartbeat
    if (this.heartbeatTimer) {
      clearTimeout(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    // Close WebSocket
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }

    // Reject pending requests
    for (const [id, request] of this.pendingRequests) {
      clearTimeout(request.timeout);
      request.reject(new Error('Connection closed'));
    }
    this.pendingRequests.clear();

    console.log('Disconnected from MCP server');
  }

  async sendRequest(request: MCPRequest): Promise<MCPResponse> {
    if (this.state !== ConnectionState.CONNECTED) {
      throw new Error('Not connected to MCP server');
    }

    const requestId = request.id || this.generateRequestId();
    request.id = requestId;

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(requestId.toString());
        reject(new Error(`Request ${requestId} timed out`));
      }, this.config.connectionTimeout);

      this.pendingRequests.set(requestId.toString(), {
        resolve,
        reject,
        timeout
      });

      // Send request
      const message = JSON.stringify(request);
      this.websocket!.send(message);
      this.lastActivity = Date.now();
    });
  }

  private setupMessageHandlers(): void {
    if (!this.websocket) return;

    this.websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const message = this.validateMessage(data);

        if (this.isResponse(message)) {
          const response = message as MCPResponse;
          const requestId = response.id?.toString();
          
          if (requestId && this.pendingRequests.has(requestId)) {
            const pending = this.pendingRequests.get(requestId)!;
            clearTimeout(pending.timeout);
            this.pendingRequests.delete(requestId);
            pending.resolve(response);
          } else {
            // Unsolicited response or notification
            this.handleNotification(response);
          }
        } else if (this.isRequest(message)) {
          // Handle server-initiated request
          this.handleServerRequest(message as MCPRequest);
        }
      } catch (error) {
        console.error('Error handling message:', error);
      }
    };

    this.websocket.onclose = () => {
      console.log('WebSocket connection closed');
      this.state = ConnectionState.DISCONNECTED;
    };

    this.websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  private async waitForWelcome(): Promise<MCPResponse | null> {
    return new Promise((resolve) => {
      const timeout = setTimeout(() => resolve(null), this.config.connectionTimeout);

      const messageHandler = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          const message = this.validateMessage(data);
          
          if (this.isResponse(message) && message.id === 'welcome') {
            clearTimeout(timeout);
            this.websocket!.removeEventListener('message', messageHandler);
            resolve(message as MCPResponse);
          }
        } catch (error) {
          // Ignore invalid messages during welcome
        }
      };

      this.websocket!.addEventListener('message', messageHandler);
    });
  }

  private async authenticate(): Promise<boolean> {
    try {
      const authData: Record<string, any> = { type: this.config.authType };

      switch (this.config.authType) {
        case 'api_key':
          authData.api_key = this.config.apiKey;
          break;
        case 'jwt':
          authData.token = this.config.jwtToken;
          break;
        case 'hmac':
          const timestamp = Math.floor(Date.now() / 1000).toString();
          authData.client_id = this.config.clientId;
          authData.timestamp = timestamp;
          authData.signature = await this.generateHmacSignature(timestamp);
          break;
      }

      const request: MCPRequest = {
        jsonrpc: '2.0',
        method: 'auth/authenticate',
        params: authData
      };

      const response = await this.sendRequest(request);

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
    const message = `${this.config.clientId}${timestamp}`;
    const encoder = new TextEncoder();
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

  private startHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearTimeout(this.heartbeatTimer);
    }

    const sendPing = async () => {
      try {
        if (this.state === ConnectionState.CONNECTED) {
          const request: MCPRequest = {
            jsonrpc: '2.0',
            method: 'ping'
          };
          await this.sendRequest(request);
          this.heartbeatTimer = setTimeout(sendPing, this.config.heartbeatInterval);
        }
      } catch (error) {
        console.warn(`Heartbeat failed: ${error}`);
        // Don't schedule next heartbeat on failure
      }
    };

    this.heartbeatTimer = setTimeout(sendPing, this.config.heartbeatInterval);
  }

  private validateMessage(data: any): MCPMessage {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid message format');
    }
    if (data.jsonrpc !== '2.0') {
      throw new Error('Invalid JSON-RPC version');
    }
    return data as MCPMessage;
  }

  private isResponse(message: MCPMessage): boolean {
    return 'result' in message || 'error' in message;
  }

  private isRequest(message: MCPMessage): boolean {
    return 'method' in message;
  }

  private handleNotification(response: MCPResponse): void {
    const handlers = this.eventHandlers.get('notification') || [];
    handlers.forEach(handler => {
      try {
        handler(response);
      } catch (error) {
        console.error('Error in notification handler:', error);
      }
    });
  }

  private handleServerRequest(request: MCPRequest): void {
    console.log(`Received server request: ${request.method}`);
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  addEventListener(event: string, handler: (data: any) => void): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)!.push(handler);
  }

  get isConnected(): boolean {
    return this.state === ConnectionState.CONNECTED;
  }

  get isIdle(): boolean {
    return Date.now() - this.lastActivity > this.config.idleTimeout!;
  }
}

class MCPConnectionPool {
  private config: MCPClientConfig;
  private connections: MCPConnection[] = [];
  private availableConnections: MCPConnection[] = [];
  private lock = false;

  constructor(config: MCPClientConfig) {
    this.config = config;
  }

  async getConnection(): Promise<MCPConnection> {
    // Try to get from available connections
    if (this.availableConnections.length > 0) {
      const connection = this.availableConnections.shift()!;
      if (connection.isConnected) {
        return connection;
      }
    }

    // Create new connection if under limit
    if (!this.lock && this.connections.length < this.config.maxConnections!) {
      this.lock = true;
      try {
        const connection = new MCPConnection(this.config);
        if (await connection.connect()) {
          this.connections.push(connection);
          return connection;
        }
      } finally {
        this.lock = false;
      }
    }

    // Wait for available connection (simplified - in real implementation would use proper queue)
    return new Promise((resolve) => {
      const checkForConnection = () => {
        if (this.availableConnections.length > 0) {
          resolve(this.availableConnections.shift()!);
        } else {
          setTimeout(checkForConnection, 100);
        }
      };
      checkForConnection();
    });
  }

  async returnConnection(connection: MCPConnection): Promise<void> {
    if (connection.isConnected && !connection.isIdle) {
      this.availableConnections.push(connection);
    } else {
      // Remove from pool
      const index = this.connections.indexOf(connection);
      if (index !== -1) {
        this.connections.splice(index, 1);
      }
      await connection.disconnect();
    }
  }

  async closeAll(): Promise<void> {
    for (const connection of this.connections) {
      await connection.disconnect();
    }
    this.connections = [];
    this.availableConnections = [];
  }
}

export class MCPClient {
  private config: MCPClientConfig;
  private connectionPool: MCPConnectionPool;
  private cache: Map<string, { result: any; timestamp: number }> = new Map();
  private availableTools: MCPTool[] = [];

  constructor(config: MCPClientConfig) {
    this.config = {
      authType: 'api_key',
      maxRetries: 3,
      retryDelay: 1000,
      connectionTimeout: 10000,
      heartbeatInterval: 30000,
      maxConnections: 5,
      idleTimeout: 300000,
      enableCaching: true,
      cacheTtl: 3600000,
      ...config
    };
    this.connectionPool = new MCPConnectionPool(this.config);
  }

  async connect(): Promise<void> {
    // Get initial connection to load capabilities
    const connection = await this.connectionPool.getConnection();
    try {
      await this.loadTools(connection);
    } finally {
      await this.connectionPool.returnConnection(connection);
    }
  }

  async disconnect(): Promise<void> {
    await this.connectionPool.closeAll();
  }

  async callTool(toolName: string, parameters: Record<string, any>, useCache = true): Promise<any> {
    // Check cache first
    if (useCache && this.config.enableCaching) {
      const cacheKey = this.getCacheKey(toolName, parameters);
      const cachedResult = this.getCachedResult(cacheKey);
      if (cachedResult) {
        return cachedResult;
      }
    }

    // Get connection and make request
    const connection = await this.connectionPool.getConnection();

    try {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        method: 'tools/call',
        params: {
          name: toolName,
          arguments: parameters
        }
      };

      const response = await this.sendWithRetry(connection, request);

      if (response.error) {
        throw new Error(`Tool call failed: ${response.error.message}`);
      }

      const result = response.result;

      // Cache result
      if (useCache && this.config.enableCaching) {
        const cacheKey = this.getCacheKey(toolName, parameters);
        this.cacheResult(cacheKey, result);
      }

      return result;
    } finally {
      await this.connectionPool.returnConnection(connection);
    }
  }

  async listTools(): Promise<MCPTool[]> {
    if (this.availableTools.length > 0) {
      return this.availableTools;
    }

    const connection = await this.connectionPool.getConnection();
    try {
      await this.loadTools(connection);
      return this.availableTools;
    } finally {
      await this.connectionPool.returnConnection(connection);
    }
  }

  async getServiceStatus(): Promise<any> {
    const connection = await this.connectionPool.getConnection();
    try {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        method: 'service/status'
      };

      const response = await this.sendWithRetry(connection, request);

      if (response.error) {
        throw new Error(`Status check failed: ${response.error.message}`);
      }

      return response.result;
    } finally {
      await this.connectionPool.returnConnection(connection);
    }
  }

  private async loadTools(connection: MCPConnection): Promise<void> {
    const request: MCPRequest = {
      jsonrpc: '2.0',
      method: 'tools/list'
    };

    const response = await this.sendWithRetry(connection, request);

    if (response.error) {
      console.error(`Failed to load tools: ${response.error.message}`);
      return;
    }

    this.availableTools = response.result as MCPTool[];
    console.log(`Loaded ${this.availableTools.length} MCP tools`);
  }

  private async sendWithRetry(connection: MCPConnection, request: MCPRequest): Promise<MCPResponse> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= this.config.maxRetries!; attempt++) {
      try {
        if (attempt > 0) {
          await this.sleep(this.config.retryDelay! * attempt);
        }

        return await connection.sendRequest(request);
      } catch (error) {
        lastError = error as Error;
        console.warn(`Request attempt ${attempt + 1} failed: ${error}`);

        // Try to reconnect if connection failed
        if (!connection.isConnected) {
          await connection.connect();
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

  private getCachedResult(cacheKey: string): any | null {
    const cached = this.cache.get(cacheKey);
    if (!cached) {
      return null;
    }

    if (Date.now() - cached.timestamp > this.config.cacheTtl!) {
      this.cache.delete(cacheKey);
      return null;
    }

    return cached.result;
  }

  private cacheResult(cacheKey: string, result: any): void {
    this.cache.set(cacheKey, {
      result,
      timestamp: Date.now()
    });

    // Simple cache cleanup - remove oldest entries if cache gets too large
    if (this.cache.size > 1000) {
      const oldestKey = Array.from(this.cache.entries())
        .sort(([,a], [,b]) => a.timestamp - b.timestamp)[0][0];
      this.cache.delete(oldestKey);
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Convenience functions
export async function createMCPClient(serverUrl: string, apiKey: string, options: Partial<MCPClientConfig> = {}): Promise<MCPClient> {
  const config: MCPClientConfig = {
    serverUrl,
    authType: 'api_key',
    apiKey,
    ...options
  };

  const client = new MCPClient(config);
  await client.connect();
  return client;
}

export async function createMCPClientJWT(serverUrl: string, jwtToken: string, options: Partial<MCPClientConfig> = {}): Promise<MCPClient> {
  const config: MCPClientConfig = {
    serverUrl,
    authType: 'jwt',
    jwtToken,
    ...options
  };

  const client = new MCPClient(config);
  await client.connect();
  return client;
}

export { MCPClient, MCPClientConfig, MCPTool, MCPRequest, MCPResponse, MCPError };
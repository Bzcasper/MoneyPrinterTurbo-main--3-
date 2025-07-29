# Vercel-Compatible TypeScript MCP Client

A fully Vercel-compatible TypeScript/JavaScript client SDK for connecting to MoneyPrinterTurbo MCP servers. This client uses HTTP/fetch instead of WebSocket for maximum compatibility with serverless environments.

## Features

- 🚀 **Vercel Serverless Compatible** - No WebSocket dependencies
- 🔒 **Multiple Authentication Methods** - API Key, JWT, and HMAC support
- 🔄 **Automatic Retry Logic** - Built-in retry with exponential backoff
- 💾 **Smart Caching** - Configurable response caching for better performance
- 🛡️ **Type Safety** - Full TypeScript support with strict typing
- 🌐 **Browser Compatible** - Works in browsers, Node.js, and serverless environments
- ⚡ **Lightweight** - Zero external dependencies

## Installation

```bash
npm install @moneyprinter/mcp-client
```

## Quick Start

### API Key Authentication

```typescript
import { createMCPClient } from '@moneyprinter/mcp-client';

const client = await createMCPClient(
  'https://your-mcp-server.vercel.app/api/mcp',
  'your-api-key'
);

// List available tools
const tools = await client.listTools();
console.log('Available tools:', tools);

// Call a tool
const result = await client.callTool('generate_video', {
  topic: 'AI and the future',
  duration: 60
});
console.log('Video generation result:', result);
```

### JWT Authentication

```typescript
import { createMCPClientJWT } from '@moneyprinter/mcp-client';

const client = await createMCPClientJWT(
  'https://your-mcp-server.vercel.app/api/mcp',
  'your-jwt-token'
);
```

### HMAC Authentication

```typescript
import { createMCPClientHMAC } from '@moneyprinter/mcp-client';

const client = await createMCPClientHMAC(
  'https://your-mcp-server.vercel.app/api/mcp',
  'your-client-id',
  'your-client-secret'
);
```

## Advanced Configuration

```typescript
import { MCPClient } from '@moneyprinter/mcp-client';

const client = new MCPClient({
  serverUrl: 'https://your-mcp-server.vercel.app/api/mcp',
  authType: 'api_key',
  apiKey: 'your-api-key',
  
  // Request settings
  maxRetries: 5,
  retryDelay: 2000,
  requestTimeout: 45000,
  
  // Caching
  enableCaching: true,
  cacheTtl: 1800000, // 30 minutes
  
  // Custom headers
  customHeaders: {
    'X-Custom-Header': 'custom-value'
  }
});

await client.connect();
```

## API Reference

### MCPClient Class

#### Constructor Options

```typescript
interface MCPClientConfig {
  serverUrl: string;                    // Required: MCP server URL
  authType: 'api_key' | 'jwt' | 'hmac'; // Required: Authentication type
  apiKey?: string;                      // For API key auth
  jwtToken?: string;                    // For JWT auth
  clientId?: string;                    // For HMAC auth
  clientSecret?: string;                // For HMAC auth
  
  // Connection settings
  maxRetries: number;                   // Default: 3
  retryDelay: number;                   // Default: 1000ms
  requestTimeout: number;               // Default: 30000ms
  
  // Cache settings
  enableCaching: boolean;               // Default: true
  cacheTtl: number;                     // Default: 3600000ms (1 hour)
  
  // Optional headers
  customHeaders?: Record<string, string>;
}
```

#### Methods

##### `connect(): Promise<void>`
Establishes connection and loads available tools.

##### `disconnect(): Promise<void>`
Closes connection and clears cache.

##### `callTool(toolName: string, parameters: Record<string, any>, useCache?: boolean): Promise<any>`
Calls an MCP tool with given parameters.

- `toolName` - Name of the tool to call
- `parameters` - Tool parameters as key-value pairs
- `useCache` - Whether to use cached results (default: true)

##### `listTools(): Promise<MCPTool[]>`
Returns list of available tools.

##### `getServiceStatus(): Promise<any>`
Gets server status information.

##### `isConnected: boolean`
Connection status property.

##### `connectionState: ConnectionState`
Current connection state.

## Usage in Vercel Functions

### API Route Example

```typescript
// pages/api/video-generate.ts
import { NextApiRequest, NextApiResponse } from 'next';
import { createMCPClient } from '@moneyprinter/mcp-client';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const client = await createMCPClient(
      process.env.MCP_SERVER_URL!,
      process.env.MCP_API_KEY!
    );

    const result = await client.callTool('generate_video', req.body);
    
    await client.disconnect();
    
    res.status(200).json(result);
  } catch (error) {
    console.error('Video generation failed:', error);
    res.status(500).json({ 
      error: 'Video generation failed',
      message: error.message 
    });
  }
}
```

### Edge Function Example

```typescript
// pages/api/edge-example.ts
import { createMCPClient } from '@moneyprinter/mcp-client';

export const config = {
  runtime: 'edge',
};

export default async function handler(request: Request) {
  try {
    const client = await createMCPClient(
      process.env.MCP_SERVER_URL!,
      process.env.MCP_API_KEY!
    );

    const body = await request.json();
    const result = await client.callTool('analyze_content', body);
    
    return new Response(JSON.stringify(result), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }), 
      { status: 500 }
    );
  }
}
```

## Error Handling

```typescript
import { MCPClient, ConnectionState } from '@moneyprinter/mcp-client';

try {
  const client = new MCPClient({
    serverUrl: 'https://your-server.com/api/mcp',
    authType: 'api_key',
    apiKey: 'your-key'
  });

  await client.connect();
  
  if (client.connectionState !== ConnectionState.CONNECTED) {
    throw new Error('Failed to connect to MCP server');
  }

  const result = await client.callTool('your_tool', { param: 'value' });
  
} catch (error) {
  if (error.message.includes('Authentication failed')) {
    console.error('Invalid credentials');
  } else if (error.message.includes('timeout')) {
    console.error('Request timed out');
  } else {
    console.error('Unexpected error:', error);
  }
}
```

## Caching

The client includes intelligent caching to improve performance:

```typescript
// Cache is enabled by default
const result1 = await client.callTool('expensive_operation', { data: 'test' });

// This will use cached result (if within TTL)
const result2 = await client.callTool('expensive_operation', { data: 'test' });

// Force bypass cache
const freshResult = await client.callTool('expensive_operation', { data: 'test' }, false);
```

## Environment Variables

Create a `.env.local` file for Vercel deployment:

```env
MCP_SERVER_URL=https://your-mcp-server.vercel.app/api/mcp
MCP_API_KEY=your-api-key
MCP_CLIENT_ID=your-client-id
MCP_CLIENT_SECRET=your-client-secret
```

## TypeScript Types

```typescript
export interface MCPTool {
  name: string;
  description: string;
  input_schema: Record<string, any>;
  output_schema?: Record<string, any>;
  category?: string;
  version?: string;
}

export interface MCPError {
  code: number;
  message: string;
  data?: Record<string, any>;
}

export enum ConnectionState {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  FAILED = 'failed'
}
```

## Migration from WebSocket Version

The HTTP-based client maintains the same API as the WebSocket version:

```typescript
// Old WebSocket client (not Vercel compatible)
import { MCPClient } from '@moneyprinter/mcp-client/websocket';

// New HTTP client (Vercel compatible)
import { MCPClient } from '@moneyprinter/mcp-client';

// API remains the same
const client = new MCPClient({ /* same config */ });
```

## Building and Development

```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Build for Vercel specifically
npm run build:vercel

# Development with watch mode
npm run dev

# Run tests
npm test

# Lint code
npm run lint
```

## Browser Compatibility

This client works in modern browsers with fetch support:

```html
<script type="module">
import { createMCPClient } from 'https://unpkg.com/@moneyprinter/mcp-client';

const client = await createMCPClient(
  'https://api.example.com/mcp',
  'your-api-key'
);
</script>
```

## Troubleshooting

### Connection Issues

1. **Check server URL** - Ensure the MCP server endpoint is correct
2. **Verify authentication** - Check API keys and credentials
3. **Network connectivity** - Test if the server is reachable
4. **CORS settings** - Ensure server allows requests from your domain

### Performance Issues

1. **Enable caching** - Use built-in caching for repeated requests
2. **Adjust timeouts** - Increase `requestTimeout` for slow operations
3. **Optimize retry logic** - Adjust `maxRetries` and `retryDelay`

### Deployment Issues

1. **Environment variables** - Ensure all required env vars are set in Vercel
2. **Function timeout** - Check Vercel function timeout limits
3. **Memory limits** - Monitor memory usage in large operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- GitHub Issues: [Report bugs](https://github.com/harry0703/MoneyPrinterTurbo/issues)
- Documentation: [Full docs](https://github.com/harry0703/MoneyPrinterTurbo#readme)
- Email: support@moneyprinterturbo.com
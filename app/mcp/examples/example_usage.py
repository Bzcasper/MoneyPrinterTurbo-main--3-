"""
MCP Integration Examples

Demonstrates how to use the MoneyPrinterTurbo MCP implementation for various
video generation tasks and AI model coordination scenarios.
"""

import asyncio
import json
from typing import Dict, Any

from app.mcp.client import MCPClient, MCPClientConfig, create_mcp_client
from app.mcp.server import MCPServer
from app.mcp.tools import mcp_tools
from app.mcp.discovery import MCPServiceRegistry


async def example_basic_client_usage():
    """Basic example of using MCP client to call video generation tools"""
    print("=== Basic MCP Client Usage ===")
    
    # Create client with API key authentication
    client = await create_mcp_client(
        server_url="ws://localhost:8081",
        api_key="default_admin"
    )
    
    try:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")
        
        # Generate a video script
        script_result = await client.call_tool(
            "generate_video_script",
            {
                "video_subject": "The wonders of space exploration",
                "language": "en",
                "paragraph_number": 2
            }
        )
        print(f"Generated script: {script_result['script'][:100]}...")
        
        # Generate video terms based on the script
        terms_result = await client.call_tool(
            "generate_video_terms",
            {
                "video_subject": "The wonders of space exploration",
                "video_script": script_result['script'],
                "amount": 5
            }
        )
        print(f"Generated terms: {terms_result['terms']}")
        
        # Create a complete video
        video_result = await client.call_tool(
            "create_video",
            {
                "video_subject": "The wonders of space exploration",
                "video_script": script_result['script'],
                "video_aspect": "16:9",
                "subtitle_enabled": True
            }
        )
        print(f"Video creation initiated: {video_result}")
        
    finally:
        await client.disconnect()


async def example_batch_processing():
    """Example of batch processing multiple video generation tasks"""
    print("\n=== Batch Processing Example ===")
    
    client = await create_mcp_client(
        server_url="ws://localhost:8081",
        api_key="default_admin"
    )
    
    try:
        # Define multiple video subjects
        subjects = [
            "Artificial Intelligence in Healthcare",
            "Renewable Energy Solutions",
            "Future of Transportation",
            "Digital Privacy and Security"
        ]
        
        # Create batch requests
        batch_requests = []
        for subject in subjects:
            batch_requests.append({
                "tool_name": "generate_video_script",
                "parameters": {
                    "video_subject": subject,
                    "language": "en",
                    "paragraph_number": 1
                }
            })
        
        # Execute batch
        print(f"Processing {len(batch_requests)} video scripts in batch...")
        start_time = asyncio.get_event_loop().time()
        
        # Using individual calls (could also use batch endpoint via REST API)
        tasks = []
        for req in batch_requests:
            task = client.call_tool(req["tool_name"], req["parameters"])
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        execution_time = asyncio.get_event_loop().time() - start_time
        print(f"Batch completed in {execution_time:.2f} seconds")
        
        # Process results
        successful = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  {subjects[i]}: ERROR - {result}")
            else:
                print(f"  {subjects[i]}: SUCCESS ({result['word_count']} words)")
                successful += 1
                
        print(f"Success rate: {successful}/{len(subjects)} ({successful/len(subjects)*100:.1f}%)")
        
    finally:
        await client.disconnect()


async def example_service_discovery():
    """Example of using service discovery for distributed MCP services"""
    print("\n=== Service Discovery Example ===")
    
    registry = MCPServiceRegistry()
    
    try:
        # Register a service
        service_id = await registry.register_service(
            "video_generator",
            "ws://localhost:8081",
            {
                "capabilities": ["video_generation", "script_generation"],
                "version": "1.0.0",
                "load_capacity": 10
            }
        )
        print(f"Registered service with ID: {service_id}")
        
        # Discover services
        services = await registry.discover_services("video_generator")
        print(f"Found {len(services)} video_generator services")
        
        # Get service instance using load balancing
        from app.mcp.discovery import LoadBalancingStrategy
        
        for strategy in [LoadBalancingStrategy.ROUND_ROBIN, LoadBalancingStrategy.RANDOM]:
            instance = await registry.get_service_instance("video_generator", strategy)
            if instance:
                print(f"Selected instance using {strategy}: {instance.endpoint}")
                
        # Get service statistics
        stats = await registry.get_service_stats("video_generator")
        print(f"Service stats: {json.dumps(stats, indent=2)}")
        
    finally:
        await registry.shutdown()


async def example_monitoring_and_alerts():
    """Example of monitoring MCP operations and handling alerts"""
    print("\n=== Monitoring and Alerts Example ===")
    
    from app.mcp.monitoring import mcp_monitor, AlertRule, track_operation
    
    # Start monitoring
    await mcp_monitor.start_monitoring()
    
    try:
        # Add custom alert rule
        custom_rule = AlertRule(
            name="high_tool_usage",
            metric="mcp.tools.calls",
            condition="> 50",
            duration=60,  # 1 minute
            cooldown=300, # 5 minutes
            severity="info"
        )
        mcp_monitor.alert_manager.add_alert_rule(custom_rule)
        
        # Add alert handler
        async def alert_handler(alert_data):
            print(f"ALERT: {alert_data['rule_name']} - {alert_data['status']}")
            print(f"  Metric: {alert_data['metric']} = {alert_data['value']}")
            print(f"  Severity: {alert_data['severity']}")
            
        mcp_monitor.alert_manager.add_alert_handler(alert_handler)
        
        # Simulate some operations to generate metrics
        @track_operation("example_operation")
        async def example_operation():
            await asyncio.sleep(0.1)  # Simulate work
            return "success"
            
        # Run operations to generate metrics
        print("Running operations to generate metrics...")
        for i in range(10):
            result = await example_operation()
            mcp_monitor.performance_tracker.track_tool_usage("example_tool", True, 0.1)
            
        # Get dashboard data
        dashboard_data = mcp_monitor.get_dashboard_data()
        print(f"Dashboard data: {json.dumps(dashboard_data, indent=2)}")
        
        # Wait a bit for potential alerts
        await asyncio.sleep(2)
        
    finally:
        await mcp_monitor.stop_monitoring()


async def example_error_handling():
    """Example of proper error handling with MCP client"""
    print("\n=== Error Handling Example ===")
    
    try:
        # Try to connect to non-existent server
        client = await create_mcp_client(
            server_url="ws://localhost:9999",  # Wrong port
            api_key="test_key",
            connection_timeout=5.0,
            max_retries=2
        )
        
    except Exception as e:
        print(f"Connection failed as expected: {e}")
        
    # Connect to real server
    client = await create_mcp_client(
        server_url="ws://localhost:8081",
        api_key="default_admin"
    )
    
    try:
        # Try to call non-existent tool
        try:
            result = await client.call_tool("non_existent_tool", {})
        except Exception as e:
            print(f"Tool call failed as expected: {e}")
            
        # Try to call tool with invalid parameters
        try:
            result = await client.call_tool("generate_video_script", {
                "invalid_param": "value"
            })
        except Exception as e:
            print(f"Invalid parameters failed as expected: {e}")
            
        # Successful call
        result = await client.call_tool("generate_video_script", {
            "video_subject": "Error handling in software",
            "paragraph_number": 1
        })
        print(f"Successful call: Generated {result['word_count']} word script")
        
    finally:
        await client.disconnect()


async def example_authentication_methods():
    """Example of different authentication methods"""
    print("\n=== Authentication Methods Example ===")
    
    # API Key authentication (already shown in other examples)
    print("Testing API key authentication...")
    try:
        client = await create_mcp_client(
            server_url="ws://localhost:8081",
            api_key="default_admin"
        )
        status = await client.get_service_status()
        print(f"API key auth successful: {status['status']}")
        await client.disconnect()
    except Exception as e:
        print(f"API key auth failed: {e}")
    
    # JWT authentication
    print("Testing JWT authentication...")
    try:
        from app.mcp.client import create_mcp_client_jwt
        from app.mcp.auth import MCPAuthenticator
        
        # Generate JWT token (in real scenario, this would come from auth service)
        authenticator = MCPAuthenticator()
        user_info = {"name": "test_user", "role": "user"}
        jwt_token = authenticator.generate_jwt_token(user_info)
        
        client = await create_mcp_client_jwt(
            server_url="ws://localhost:8081",
            jwt_token=jwt_token
        )
        status = await client.get_service_status()
        print(f"JWT auth successful: {status['status']}")
        await client.disconnect()
    except Exception as e:
        print(f"JWT auth failed: {e}")


async def main():
    """Run all examples"""
    print("MoneyPrinterTurbo MCP Integration Examples")
    print("=" * 50)
    
    examples = [
        example_basic_client_usage,
        example_batch_processing,
        example_service_discovery,
        example_monitoring_and_alerts,
        example_error_handling,
        example_authentication_methods
    ]
    
    for example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"Example {example_func.__name__} failed: {e}")
        
        print()  # Add spacing between examples


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
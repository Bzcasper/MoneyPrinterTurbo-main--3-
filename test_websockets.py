import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")
print(f"sys.path: {sys.path}")

try:
    import websockets
    print(f"Websockets version: {websockets.__version__}")
except ImportError as e:
    print(f"Error importing websockets: {e}")

print("Script completed")

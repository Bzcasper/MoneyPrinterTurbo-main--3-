"""
YouTube API Setup and Configuration Script
Helps users configure YouTube Data API v3 for automated publishing
"""

import os
import json
import sys
from typing import Dict, Any
from pathlib import Path

def create_youtube_credentials_template():
    """Create template for YouTube API credentials"""
    template = {
        "web": {
            "client_id": "YOUR_CLIENT_ID.googleusercontent.com",
            "project_id": "your-project-id",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "YOUR_CLIENT_SECRET",
            "redirect_uris": [
                "http://localhost:8080"
            ]
        }
    }
    return template

def setup_youtube_config():
    """Interactive setup for YouTube API configuration"""
    print("🎥 YouTube Shorts Publisher Configuration Setup")
    print("=" * 50)
    
    # Create config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    credentials_path = config_dir / "youtube_credentials.json"
    
    if credentials_path.exists():
        print(f"✅ YouTube credentials already exist at: {credentials_path}")
        overwrite = input("Do you want to overwrite? (y/N): ").lower().strip()
        if overwrite != 'y':
            return
    
    print("\n📋 To set up YouTube API access, you need to:")
    print("1. Go to Google Cloud Console (https://console.cloud.google.com/)")
    print("2. Create a new project or select existing one")
    print("3. Enable YouTube Data API v3")
    print("4. Create OAuth 2.0 Client IDs credentials")
    print("5. Download the JSON file")
    print("\n🔍 Detailed instructions:")
    print_setup_instructions()
    
    print("\n" + "="*50)
    
    # Get user input for credentials
    client_id = input("Enter your Client ID: ").strip()
    client_secret = input("Enter your Client Secret: ").strip()
    project_id = input("Enter your Project ID: ").strip()
    
    if not all([client_id, client_secret, project_id]):
        print("❌ All fields are required. Setup cancelled.")
        return
    
    # Create credentials file
    credentials_data = {
        "web": {
            "client_id": client_id,
            "project_id": project_id,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost"]
        }
    }
    
    try:
        with open(credentials_path, 'w') as f:
            json.dump(credentials_data, f, indent=2)
        
        print(f"✅ Credentials saved to: {credentials_path}")
        print("\n🚀 Next steps:")
        print("1. Run your first upload to complete OAuth flow")
        print("2. Test with: python -m app.services.youtube_publisher")
        print("3. Integration is ready for automated publishing!")
        
        # Create example usage file
        create_usage_example()
        
    except Exception as e:
        print(f"❌ Failed to save credentials: {e}")

def print_setup_instructions():
    """Print detailed setup instructions"""
    instructions = """
📚 DETAILED SETUP INSTRUCTIONS:

1. 🌐 Google Cloud Console Setup:
   - Visit: https://console.cloud.google.com/
   - Create new project or select existing
   - Project name suggestion: "MoneyPrinterTurbo-YouTube"

2. 🔌 Enable YouTube Data API:
   - Navigate to: APIs & Services > Library
   - Search: "YouTube Data API v3"
   - Click "ENABLE"

3. 🔐 Create OAuth 2.0 Credentials:
   - Go to: APIs & Services > Credentials  
   - Click: "+ CREATE CREDENTIALS" > "OAuth client ID"
   - Application type: "Desktop application"
   - Name: "YouTube Publisher"
   - Download JSON file

4. 📋 Copy Information:
   From the downloaded JSON file, you'll need:
   - client_id (ends with .googleusercontent.com)
   - client_secret (random string)
   - project_id (your project identifier)

5. ⚡ OAuth Consent Screen (if required):
   - Configure OAuth consent screen
   - Add scopes: ../auth/youtube.upload
   - Add test users (your account)

💡 TIPS:
- Keep credentials secure and never commit to version control
- The first run will open browser for authorization
- Tokens are stored locally for subsequent runs
"""
    print(instructions)

def create_usage_example():
    """Create example usage file"""
    example_code = '''"""
YouTube Shorts Publisher Usage Example
Run this after completing setup to test the integration
"""

from app.services.youtube_publisher import YouTubeShortsPublisher

def test_youtube_integration():
    """Test YouTube API integration"""
    try:
        print("🧪 Testing YouTube API integration...")
        
        # Initialize publisher (will trigger OAuth flow on first run)
        publisher = YouTubeShortsPublisher()
        
        print("✅ YouTube API authentication successful!")
        
        # Test metadata optimization
        test_content = "成功的秘诀在于永不放弃，坚持到最后一刻"
        test_keywords = ["成功", "励志", "坚持", "正能量"]
        
        title = publisher.optimize_title(test_content, "成功")
        description = publisher.generate_description(test_content, test_keywords, "成功")
        
        print(f"\\n📝 Test Optimization Results:")
        print(f"Title: {title}")
        print(f"Description Preview: {description[:100]}...")
        
        # Test thumbnail creation
        thumbnail_path = "test_thumbnail.jpg"
        publisher.create_thumbnail(title, thumbnail_path)
        print(f"🖼️  Test thumbnail created: {thumbnail_path}")
        
        print("\\n🎉 All tests passed! Ready for video publishing.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\\n🔧 Troubleshooting:")
        print("1. Check your credentials in config/youtube_credentials.json")
        print("2. Ensure YouTube Data API v3 is enabled")
        print("3. Complete OAuth consent screen setup if required")

if __name__ == "__main__":
    test_youtube_integration()
'''
    
    example_path = Path("test_youtube_integration.py")
    try:
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write(example_code)
        print(f"📝 Usage example created: {example_path}")
    except Exception as e:
        print(f"⚠️  Could not create usage example: {e}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "google-api-python-client",
        "google-auth-httplib2", 
        "google-auth-oauthlib",
        "Pillow"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print(f"\n📦 Install with: pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def main():
    """Main setup function"""
    print("🎥 YouTube Shorts Publisher Setup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n🛑 Please install missing dependencies first")
        return
    
    # Run interactive setup
    setup_youtube_config()

if __name__ == "__main__":
    main()
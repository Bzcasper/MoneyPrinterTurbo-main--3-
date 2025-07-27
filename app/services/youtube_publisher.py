"""
YouTube Shorts Publishing Service
Optimized for Chinese content and automated upload workflow
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import random

import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from PIL import Image, ImageDraw, ImageFont
import requests

logger = logging.getLogger(__name__)

class YouTubeShortsPublisher:
    """
    Automated YouTube Shorts publishing service with Chinese content optimization
    """
    
    # YouTube API scopes
    SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
    
    def __init__(self, credentials_path: str = "youtube_credentials.json", 
                 token_path: str = "youtube_token.json"):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self._authenticate()
        
        # Chinese content optimization templates
        self.title_templates = [
            "🔥 {topic} | 30秒改变你的思维 #Shorts",
            "💪 {topic} - 每日正能量分享 #励志", 
            "✨ {topic} | 值得收藏的人生感悟",
            "🚀 {topic} - 成功人士都在做的事",
            "⚡ {topic} | 一句话点醒你 #正能量",
            "🌟 {topic} - 改变命运从这里开始"
        ]
        
        self.description_template = """
{hook}

{main_content}

📌 关键词: {keywords}
🏷️ {hashtags}

💪 每日正能量，关注获取更多励志内容！
🔔 开启小铃铛，不错过精彩视频

👍 点赞支持 | 📢 分享给朋友 | 💬 评论互动
"""
        
        # Optimized hashtags for Chinese audience
        self.base_hashtags = ["#Shorts", "#励志", "#正能量", "#成功", "#自我提升"]
        self.topic_hashtags = {
            "职场": ["#职场", "#工作", "#升职", "#职业规划"],
            "创业": ["#创业", "#商业", "#赚钱", "#投资"],
            "学习": ["#学习", "#读书", "#知识", "#成长"],
            "生活": ["#生活", "#人生", "#感悟", "#智慧"],
            "心态": ["#心态", "#思维", "#格局", "#心理"]
        }
        
    def _authenticate(self) -> None:
        """Authenticate with YouTube API using OAuth 2.0"""
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)
        
        # If no valid credentials, start OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"YouTube API credentials not found at {self.credentials_path}. "
                        "Please download from Google Cloud Console."
                    )
                    
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('youtube', 'v3', credentials=creds)
        logger.info("✅ YouTube API authentication successful")
    
    def optimize_title(self, content: str, topic_category: str = "general") -> str:
        """
        Generate optimized title for YouTube Shorts with Chinese SEO
        
        Args:
            content: Main content/topic
            topic_category: Category for targeted hashtags
            
        Returns:
            Optimized title string
        """
        # Extract key phrases from content (simplified)
        key_phrase = content[:20] + "..." if len(content) > 20 else content
        
        # Select random template
        template = random.choice(self.title_templates)
        optimized_title = template.format(topic=key_phrase)
        
        # Ensure title length is optimal (under 60 characters for mobile)
        if len(optimized_title) > 55:
            optimized_title = optimized_title[:52] + "..."
            
        logger.info(f"📝 Optimized title: {optimized_title}")
        return optimized_title
    
    def generate_description(self, content: str, keywords: List[str], 
                           topic_category: str = "general") -> str:
        """
        Generate SEO-optimized description for YouTube Shorts
        
        Args:
            content: Main video content
            keywords: List of keywords for SEO
            topic_category: Category for targeted hashtags
            
        Returns:
            Optimized description string
        """
        # Generate hook
        hooks = [
            "你知道吗？",
            "成功的秘密在于...",
            "改变命运的关键是...",
            "这个道理很多人不懂：",
            "听完这段话，你就明白了："
        ]
        hook = random.choice(hooks)
        
        # Combine hashtags
        hashtags = self.base_hashtags.copy()
        if topic_category in self.topic_hashtags:
            hashtags.extend(self.topic_hashtags[topic_category])
        
        # Add trending hashtags based on time
        current_time = datetime.now()
        if current_time.weekday() == 0:  # Monday
            hashtags.append("#周一正能量")
        elif current_time.hour >= 21:  # Evening
            hashtags.append("#晚安心语")
            
        hashtags_text = " ".join(hashtags)
        keywords_text = ", ".join(keywords)
        
        description = self.description_template.format(
            hook=hook,
            main_content=content,
            keywords=keywords_text,
            hashtags=hashtags_text
        ).strip()
        
        logger.info(f"📄 Generated description with {len(hashtags)} hashtags")
        return description
    
    def create_thumbnail(self, title: str, output_path: str, 
                        background_color: str = "#FF6B35") -> str:
        """
        Generate attractive thumbnail for YouTube Shorts
        
        Args:
            title: Video title text
            output_path: Path to save thumbnail
            background_color: Background color hex code
            
        Returns:
            Path to generated thumbnail
        """
        # Thumbnail dimensions for YouTube
        width, height = 1280, 720
        
        # Create image
        image = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(image)
        
        # Try to load Chinese font, fallback to default
        try:
            font_path = "/home/bobby/Documents/MoneyPrinterTurbo/resource/fonts/MicrosoftYaHeiBold.ttc"
            if os.path.exists(font_path):
                font_large = ImageFont.truetype(font_path, 80)
                font_small = ImageFont.truetype(font_path, 40)
            else:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Clean title for thumbnail
        clean_title = re.sub(r'[🔥💪✨🚀⚡🌟#]', '', title).strip()
        
        # Add text with outline effect
        text_color = "white"
        outline_color = "black"
        
        # Center the text
        text_bbox = draw.textbbox((0, 0), clean_title, font=font_large)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2 - 50
        
        # Draw outline
        for adj_x in range(-2, 3):
            for adj_y in range(-2, 3):
                if adj_x != 0 or adj_y != 0:
                    draw.text((x + adj_x, y + adj_y), clean_title, 
                             font=font_large, fill=outline_color)
        
        # Draw main text
        draw.text((x, y), clean_title, font=font_large, fill=text_color)
        
        # Add "SHORTS" label
        shorts_text = "SHORTS"
        shorts_bbox = draw.textbbox((0, 0), shorts_text, font=font_small)
        shorts_width = shorts_bbox[2] - shorts_bbox[0]
        
        draw.rectangle((width - shorts_width - 40, 30, width - 20, 80), 
                      fill="red")
        draw.text((width - shorts_width - 30, 40), shorts_text, 
                 font=font_small, fill="white")
        
        # Save thumbnail
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, "JPEG", quality=95)
        
        logger.info(f"🖼️ Thumbnail created: {output_path}")
        return output_path
    
    def upload_video(self, video_path: str, title: str, description: str, 
                    tags: List[str], thumbnail_path: Optional[str] = None,
                    privacy_status: str = "public") -> Dict:
        """
        Upload video to YouTube as Shorts
        
        Args:
            video_path: Path to video file
            title: Video title
            description: Video description
            tags: List of tags
            thumbnail_path: Optional thumbnail image path
            privacy_status: Video privacy (public, private, unlisted)
            
        Returns:
            Upload result dictionary
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Validate video for Shorts requirements
        file_size = os.path.getsize(video_path)
        if file_size > 256 * 1024 * 1024:  # 256MB limit
            logger.warning(f"⚠️ Video file size ({file_size / 1024 / 1024:.1f}MB) may exceed Shorts limits")
        
        # Prepare upload metadata
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
                'categoryId': '22',  # People & Blogs category
                'defaultLanguage': 'zh-CN',
                'defaultAudioLanguage': 'zh-CN'
            },
            'status': {
                'privacyStatus': privacy_status,
                'selfDeclaredMadeForKids': False
            }
        }
        
        # Create media upload object
        media = MediaFileUpload(
            video_path,
            chunksize=-1,
            resumable=True,
            mimetype='video/*'
        )
        
        try:
            logger.info(f"📤 Starting upload: {os.path.basename(video_path)}")
            
            # Execute upload
            insert_request = self.service.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            # Upload with progress tracking
            response = self._resumable_upload(insert_request)
            
            video_id = response['id']
            video_url = f"https://youtu.be/{video_id}"
            
            logger.info(f"✅ Upload successful: {video_url}")
            
            # Upload thumbnail if provided
            if thumbnail_path and os.path.exists(thumbnail_path):
                self._upload_thumbnail(video_id, thumbnail_path)
            
            # Store upload info in coordination memory
            self._store_upload_result(video_id, title, video_url)
            
            return {
                'success': True,
                'video_id': video_id,
                'video_url': video_url,
                'title': title,
                'upload_time': datetime.now(timezone.utc).isoformat()
            }
            
        except HttpError as e:
            error_details = json.loads(e.content.decode('utf-8'))
            error_msg = error_details.get('error', {}).get('message', str(e))
            logger.error(f"❌ Upload failed: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'details': error_details
            }
    
    def _resumable_upload(self, insert_request):
        """Execute resumable upload with progress tracking"""
        response = None
        error = None
        retry = 0
        
        while response is None:
            try:
                status, response = insert_request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"📊 Upload progress: {progress}%")
                    
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    # Retriable HTTP errors
                    error = f"Retriable error occurred: {e.resp.status}"
                    logger.warning(error)
                    
                    retry += 1
                    if retry > 3:
                        logger.error("Max retries exceeded")
                        raise
                        
                    time.sleep(2 ** retry)  # Exponential backoff
                else:
                    raise
                    
        return response
    
    def _upload_thumbnail(self, video_id: str, thumbnail_path: str) -> bool:
        """Upload custom thumbnail for video"""
        try:
            media = MediaFileUpload(thumbnail_path, mimetype='image/*')
            self.service.thumbnails().set(
                videoId=video_id,
                media_body=media
            ).execute()
            
            logger.info(f"🖼️ Thumbnail uploaded for video: {video_id}")
            return True
            
        except HttpError as e:
            logger.error(f"❌ Thumbnail upload failed: {e}")
            return False
    
    def _store_upload_result(self, video_id: str, title: str, url: str):
        """Store upload result in coordination memory"""
        try:
            import subprocess
            result_data = {
                'video_id': video_id,
                'title': title,
                'url': url,
                'upload_time': datetime.now().isoformat(),
                'platform': 'youtube_shorts'
            }
            
            subprocess.run([
                'npx', 'claude-flow@alpha', 'hooks', 'notification',
                '--message', f'YouTube Shorts uploaded: {title} -> {url}',
                '--data', json.dumps(result_data)
            ], capture_output=True, timeout=10)
            
        except Exception as e:
            logger.warning(f"Failed to store coordination data: {e}")
    
    def get_optimal_upload_time(self) -> Tuple[int, int]:
        """
        Get optimal upload time for Chinese audience
        
        Returns:
            Tuple of (hour, minute) in Beijing time
        """
        # Peak engagement times for Chinese audience
        optimal_times = [
            (19, 0),   # 7 PM - evening commute
            (20, 30),  # 8:30 PM - prime time
            (21, 15),  # 9:15 PM - relaxation time
            (12, 0),   # 12 PM - lunch break
            (15, 30)   # 3:30 PM - afternoon break
        ]
        
        return random.choice(optimal_times)
    
    def schedule_upload(self, video_path: str, content: str, keywords: List[str],
                       topic_category: str = "general") -> Dict:
        """
        Complete automated upload workflow for YouTube Shorts
        
        Args:
            video_path: Path to video file
            content: Video content description
            keywords: SEO keywords
            topic_category: Content category for optimization
            
        Returns:
            Complete upload result
        """
        logger.info("🚀 Starting automated YouTube Shorts publishing workflow")
        
        try:
            # 1. Optimize title and description
            title = self.optimize_title(content, topic_category)
            description = self.generate_description(content, keywords, topic_category)
            
            # 2. Prepare tags
            tags = keywords + ["shorts", "短视频", "励志", "正能量"]
            tags = tags[:15]  # YouTube limit
            
            # 3. Create thumbnail
            thumbnail_dir = os.path.join(os.path.dirname(video_path), "thumbnails")
            thumbnail_path = os.path.join(thumbnail_dir, f"thumb_{int(time.time())}.jpg")
            self.create_thumbnail(title, thumbnail_path)
            
            # 4. Upload video
            upload_result = self.upload_video(
                video_path=video_path,
                title=title,
                description=description,
                tags=tags,
                thumbnail_path=thumbnail_path,
                privacy_status="public"
            )
            
            # 5. Performance tracking setup
            if upload_result['success']:
                logger.info("📊 Setting up performance tracking...")
                self._setup_performance_tracking(upload_result['video_id'])
            
            return upload_result
            
        except Exception as e:
            logger.error(f"❌ Automated upload workflow failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _setup_performance_tracking(self, video_id: str):
        """Setup performance tracking for uploaded video"""
        try:
            # Store video ID for future analytics
            tracking_data = {
                'video_id': video_id,
                'upload_date': datetime.now().isoformat(),
                'tracking_enabled': True
            }
            
            # Store in coordination memory for analytics agent
            import subprocess
            subprocess.run([
                'npx', 'claude-flow@alpha', 'memory', 'store',
                '--key', f'analytics/videos/{video_id}',
                '--value', json.dumps(tracking_data)
            ], capture_output=True, timeout=10)
            
            logger.info(f"📊 Performance tracking enabled for: {video_id}")
            
        except Exception as e:
            logger.warning(f"Performance tracking setup failed: {e}")

# Usage example and integration helpers
def create_publisher_config() -> Dict:
    """Create default configuration for YouTube publisher"""
    return {
        'credentials_path': 'config/youtube_credentials.json',
        'token_path': 'config/youtube_token.json',
        'upload_settings': {
            'privacy_status': 'public',
            'category_id': '22',
            'language': 'zh-CN'
        },
        'optimization': {
            'title_max_length': 55,
            'hashtags_max': 12,
            'thumbnail_size': (1280, 720)
        }
    }

if __name__ == "__main__":
    # Example usage
    publisher = YouTubeShortsPublisher()
    
    result = publisher.schedule_upload(
        video_path="/path/to/motivational_video.mp4",
        content="成功的秘诀就是永不放弃，坚持到最后一刻",
        keywords=["成功", "励志", "坚持", "正能量"],
        topic_category="成功"
    )
    
    print(f"Upload result: {result}")
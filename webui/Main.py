import os
import sys
from datetime import datetime

# No replacement needed
from loguru import logger
import requests

# Add the root directory of the project to the system path to allow importing modules from the project
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    print("******** sys.path ********")
    print(sys.path)
    print("")

# Local imports
from app.config import config
from app.models.schema import (
    VideoAspect,
    VideoConcatMode,
    VideoTransitionMode,
    VideoParams,
)
from app.services import llm
from app.services import task as tm
from app.utils import utils

st.set_page_config(
    page_title="MoneyPrinterTurbo Enhanced",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Report a bug": "https://github.com/harry0703/MoneyPrinterTurbo/issues",
        "About": "# MoneyPrinterTurbo Enhanced\nEnhanced version with additional features including:\n- Health monitoring\n- Advanced video settings\n- Batch processing\n- Template management\n- Real-time progress tracking\n\nhttps://github.com/harry0703/MoneyPrinterTurbo",
    },
)

# Enhanced CSS styling
streamlit_style = """
<style>
h1 {
    padding-top: 0 !important;
}
.stProgress .st-bo {
    background-color: #1f77b4;
}
.task-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
}
.success-card {
    border-color: #28a745;
    background-color: #f8fff9;
}
.error-card {
    border-color: #dc3545;
    background-color: #fff8f8;
}
</style>
"""
st.markdown(streamlit_style, unsafe_allow_html=True)

# 定义资源目录
font_dir = os.path.join(root_dir, "resource", "fonts")
song_dir = os.path.join(root_dir, "resource", "songs")
i18n_dir = os.path.join(root_dir, "webui", "i18n")
config_file = os.path.join(root_dir, "webui", ".streamlit", "webui.toml")
system_locale = utils.get_system_locale()

# Initialize session state with enhanced defaults
if "video_subject" not in st.session_state:
    st.session_state["video_subject"] = ""
if "video_script" not in st.session_state:
    st.session_state["video_script"] = ""
if "video_terms" not in st.session_state:
    st.session_state["video_terms"] = ""
if "ui_language" not in st.session_state:
    st.session_state["ui_language"] = config.ui.get("language", system_locale)
if "tasks" not in st.session_state:
    st.session_state["tasks"] = []
if "templates" not in st.session_state:
    st.session_state["templates"] = {}

# 加载语言文件
locales = utils.load_locales(i18n_dir)

# Enhanced sidebar with health status
with st.sidebar:
    st.title("🤖 MoneyPrinterTurbo")
    
    # Health check
    try:
        # Use service name 'api' instead of localhost for Docker container communication
        health_response = requests.get("http://api:8080/ping", timeout=2)
        if health_response.status_code == 200:
            st.success("✅ Service Healthy")
            health_data = health_response.json()
            st.metric("CPU", f"{health_data.get('system', {}).get('cpu_percent', 0)}%")
            st.metric("Memory", f"{health_data.get('system', {}).get('memory_percent', 0)}%")
        else:
            st.error("❌ Service Unhealthy")
    except Exception as e:
        logger.warning(f"Service check failed: {str(e)}")
        st.warning("⚠️ Service Check Failed")

    # Language selector
    display_languages = []
    selected_index = 0
    for i, code in enumerate(locales.keys()):
        display_languages.append(f"{code} - {locales[code].get('Language')}")
        if code == st.session_state.get("ui_language", ""):
            selected_index = i

    selected_language = st.selectbox(
        "Language / 语言",
        options=display_languages,
        index=selected_index,
        key="sidebar_language_selector",
    )
    if selected_language:
        code = selected_language.split(" - ")[0].strip()
        st.session_state["ui_language"] = code
        config.ui["language"] = code

    # Quick actions
    st.subheader("Quick Actions")
    if st.button("🗑️ Clear All Tasks"):
        st.session_state["tasks"] = []
        st.rerun()
    
    if st.button("💾 Save Template", key='save_template_quick'):
        template_name = st.text_input("Template Name")
        if template_name:
            st.session_state["templates"][template_name] = {
                "video_subject": st.session_state["video_subject"],
                "video_script": st.session_state["video_script"],
                "video_terms": st.session_state["video_terms"],
            }
            st.success(f"Template '{template_name}' saved!")

# Create log container in the main area
log_container = st.container()

# Enhanced task management
def display_task_history():
    """Display historical tasks with status."""
    if st.session_state["tasks"]:
        st.subheader("📋 Task History")
        for task in st.session_state["tasks"]:
            with st.expander(f"Task: {task['id'][:8]} - {task['status']}"):
                st.json(task)

# Add to the main interface
with st.sidebar:
    if st.checkbox("Show Task History"):
        display_task_history()

# Enhanced error handling and logging
def enhanced_log_received(msg):
    """Enhanced logging with timestamps and levels."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    
    if config.ui["hide_log"]:
        return
    
    # Store in session state for persistence
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    
    st.session_state["logs"].append(formatted_msg)
    
    # Keep only last 100 logs
    if len(st.session_state["logs"]) > 100:
        st.session_state["logs"] = st.session_state["logs"][-100:]
    
    with log_container:
        st.code("\n".join(st.session_state["logs"][-20:]))  # Show last 20

# Replace the original log_received function with enhanced version
log_received = enhanced_log_received

# Get current locale for UI translations
locale_text = locales.get(st.session_state.get("ui_language", "en"), {}).get("Translation", {})

# Main UI Layout
st.title("🤖 MoneyPrinterTurbo Enhanced")
st.markdown("---")

# Create main columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Video Script Section
    with st.expander(locale_text.get("Video Script Settings", "**Video Script Settings**"), expanded=True):
        # Video Subject Input
        video_subject = st.text_input(
            locale_text.get("Video Subject", "Video Subject"),
            value=st.session_state.get("video_subject", ""),
            key="video_subject_input",
            help=locale_text.get("Video Subject", "Provide a keyword, AI will automatically generate video script")
        )
        st.session_state["video_subject"] = video_subject

        # Script Language Selection
        script_language = st.selectbox(
            locale_text.get("Script Language", "Script Language"),
            options=["Auto Detect", "English", "Chinese", "Spanish", "French", "German", "Japanese", "Korean"],
            key="script_language_select"
        )

        # Generate Script and Keywords Button
        col_gen1, col_gen2 = st.columns(2)
        with col_gen1:
            if st.button(locale_text.get("Generate Video Script and Keywords", "🎬 Generate Script & Keywords"), 
                        type="primary", use_container_width=True):
                if video_subject:
                    with st.spinner(locale_text.get("Generating Video Script and Keywords", "Generating...")):
                        try:
                            # Generate script using LLM service
                            from app.services import llm
                            generated_script = llm.generate_script(
                                video_subject=video_subject,
                                language=script_language.lower() if script_language != "Auto Detect" else "auto",
                                paragraph_number=3
                            )
                            if generated_script:
                                st.session_state["video_script"] = generated_script
                                st.success("✅ Script generated successfully!")
                                
                                # Generate keywords
                                generated_terms = llm.generate_terms(
                                    video_subject=video_subject,
                                    video_script=generated_script,
                                    amount=5
                                )
                                if generated_terms:
                                    st.session_state["video_terms"] = ", ".join(generated_terms)
                                    st.success("✅ Keywords generated successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to generate script")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning(locale_text.get("Please Enter the Video Subject", "Please enter video subject first"))

        with col_gen2:
            if st.button(locale_text.get("Generate Video Keywords", "🔍 Generate Keywords Only"), 
                        use_container_width=True):
                if st.session_state.get("video_script"):
                    with st.spinner(locale_text.get("Generating Video Keywords", "Generating keywords...")):
                        try:
                            from app.services import llm
                            generated_terms = llm.generate_terms(
                                video_subject=video_subject,
                                video_script=st.session_state["video_script"],
                                amount=5
                            )
                            if generated_terms:
                                st.session_state["video_terms"] = ", ".join(generated_terms)
                                st.success("✅ Keywords generated!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("Please generate or enter a video script first")

        # Video Script Text Area
        video_script = st.text_area(
            locale_text.get("Video Script", "Video Script"),
            value=st.session_state.get("video_script", ""),
            height=150,
            key="video_script_input",
            help="Optional, AI generated. Proper punctuation helps with subtitle generation"
        )
        st.session_state["video_script"] = video_script

        # Video Keywords Input
        video_terms = st.text_input(
            locale_text.get("Video Keywords", "Video Keywords"),
            value=st.session_state.get("video_terms", ""),
            key="video_terms_input",
            help="Optional, AI generated. Use English commas for separation, English only"
        )
        st.session_state["video_terms"] = video_terms

    # Video Settings Section
    with st.expander(locale_text.get("Video Settings", "**Video Settings**"), expanded=True):
        col_v1, col_v2, col_v3 = st.columns(3)
        
        with col_v1:
            video_concat_mode = st.selectbox(
                locale_text.get("Video Concat Mode", "Video Concatenation Mode"),
                options=[("random", locale_text.get("Random", "Random Concatenation")), 
                        ("sequential", locale_text.get("Sequential", "Sequential Concatenation"))],
                format_func=lambda x: x[1],
                key="video_concat_mode"
            )
            
            video_aspect = st.selectbox(
                locale_text.get("Video Ratio", "Video Aspect Ratio"),
                options=[("9:16", locale_text.get("Portrait", "Portrait 9:16")), 
                        ("16:9", locale_text.get("Landscape", "Landscape 16:9"))],
                format_func=lambda x: x[1],
                key="video_aspect"
            )

        with col_v2:
            video_transition = st.selectbox(
                locale_text.get("Video Transition Mode", "Video Transition Mode"),
                options=[
                    (None, locale_text.get("None", "None")),
                    ("Shuffle", locale_text.get("Shuffle", "Shuffle")),
                    ("FadeIn", locale_text.get("FadeIn", "FadeIn")),
                    ("FadeOut", locale_text.get("FadeOut", "FadeOut")),
                    ("SlideIn", locale_text.get("SlideIn", "SlideIn")),
                    ("SlideOut", locale_text.get("SlideOut", "SlideOut"))
                ],
                format_func=lambda x: x[1],
                key="video_transition"
            )
            
            clip_duration = st.number_input(
                locale_text.get("Clip Duration", "Maximum Duration of Video Clips (seconds)"),
                min_value=1,
                max_value=10,
                value=5,
                key="clip_duration"
            )

        with col_v3:
            video_count = st.number_input(
                locale_text.get("Number of Videos Generated Simultaneously", "Number of Videos Generated"),
                min_value=1,
                max_value=5,
                value=1,
                key="video_count"
            )

    # Audio Settings Section  
    with st.expander(locale_text.get("Audio Settings", "**Audio Settings**"), expanded=True):
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            # TTS Provider Selection
            tts_provider = st.selectbox(
                locale_text.get("TTS Provider", "TTS Provider"),
                options=["azure", "edge", "openai"],
                key="tts_provider"
            )
            
            if tts_provider == "azure":
                speech_region = st.text_input(
                    locale_text.get("Speech Region", "Speech Region"),
                    key="speech_region",
                    help="Required, get from Azure portal"
                )
                speech_key = st.text_input(
                    locale_text.get("Speech Key", "API Key"),
                    type="password",
                    key="speech_key",
                    help="Required, get from Azure portal"
                )
            
            # Voice Selection
            voice_name = st.selectbox(
                locale_text.get("Speech Synthesis", "Speech Synthesis Voice"),
                options=["en-US-JennyNeural", "en-US-GuyNeural", "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
                key="voice_name"
            )

        with col_a2:
            voice_rate = st.slider(
                locale_text.get("Speech Rate", "Speech Rate"),
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key="voice_rate"
            )
            
            voice_volume = st.slider(
                locale_text.get("Speech Volume", "Speech Volume"),
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key="voice_volume"
            )
            
            # Background Music Settings
            bgm_type = st.selectbox(
                locale_text.get("Background Music", "Background Music"),
                options=[
                    ("none", locale_text.get("No Background Music", "No Background Music")),
                    ("random", locale_text.get("Random Background Music", "Random Background Music")),
                    ("custom", locale_text.get("Custom Background Music", "Custom Background Music"))
                ],
                format_func=lambda x: x[1],
                key="bgm_type"
            )
            
            if bgm_type[0] == "custom":
                bgm_file = st.text_input(
                    locale_text.get("Custom Background Music File", "Custom BGM File Path"),
                    key="bgm_file"
                )
            
            bgm_volume = st.slider(
                locale_text.get("Background Music Volume", "Background Music Volume"),
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key="bgm_volume"
            )

    # Subtitle Settings Section
    with st.expander(locale_text.get("Subtitle Settings", "**Subtitle Settings**")):
        subtitle_enabled = st.checkbox(
            locale_text.get("Enable Subtitles", "Enable Subtitles"),
            value=True,
            key="subtitle_enabled"
        )
        
        if subtitle_enabled:
            col_s1, col_s2, col_s3 = st.columns(3)
            
            with col_s1:
                subtitle_font = st.selectbox(
                    locale_text.get("Font", "Subtitle Font"),
                    options=["Arial", "Helvetica", "Times New Roman", "Courier New"],
                    key="subtitle_font"
                )
                
                subtitle_position = st.selectbox(
                    locale_text.get("Position", "Subtitle Position"),
                    options=[
                        ("top", locale_text.get("Top", "Top")),
                        ("center", locale_text.get("Center", "Center")),
                        ("bottom", locale_text.get("Bottom", "Bottom"))
                    ],
                    format_func=lambda x: x[1],
                    index=2,
                    key="subtitle_position"
                )

            with col_s2:
                subtitle_font_size = st.number_input(
                    locale_text.get("Font Size", "Font Size"),
                    min_value=12,
                    max_value=72,
                    value=24,
                    key="subtitle_font_size"
                )
                
                subtitle_font_color = st.color_picker(
                    locale_text.get("Font Color", "Font Color"),
                    value="#FFFFFF",
                    key="subtitle_font_color"
                )

            with col_s3:
                subtitle_stroke_color = st.color_picker(
                    locale_text.get("Stroke Color", "Stroke Color"),
                    value="#000000",
                    key="subtitle_stroke_color"
                )
                
                subtitle_stroke_width = st.number_input(
                    locale_text.get("Stroke Width", "Stroke Width"),
                    min_value=0,
                    max_value=10,
                    value=2,
                    key="subtitle_stroke_width"
                )

with col2:
    # Progress and Status Section
    st.subheader("📊 Generation Status")
    
    # Progress Bar (will be updated during generation)
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Task Queue Display
    if st.session_state.get("tasks"):
        st.subheader("🔄 Active Tasks")
        for task in st.session_state["tasks"][-3:]:  # Show last 3 tasks
            with st.container():
                status_color = {"pending": "🟡", "running": "🔵", "completed": "✅", "failed": "❌"}
                st.write(f"{status_color.get(task.get('status', 'pending'), '⚪')} {task.get('id', 'Unknown')[:8]}")

    # Generation Button
    st.markdown("---")
    generate_button = st.button(
        locale_text.get("Generate Video", "🎬 Generate Video"),
        type="primary",
        use_container_width=True,
        disabled=not (video_subject or video_script)
    )

    if generate_button:
        if not video_subject and not video_script:
            st.error(locale_text.get("Video Script and Subject Cannot Both Be Empty", 
                                   "Video Subject and Script cannot both be empty"))
        else:
            # Start video generation process
            with st.spinner(locale_text.get("Generating Video", "Generating video, please wait...")):
                try:
                    from app.services import task as tm
                    from app.models.schema import VideoParams, VideoConcatMode, VideoAspect, VideoTransitionMode
                    from uuid import uuid4
                    
                    # Create task ID
                    task_id = str(uuid4())
                    
                    # Create VideoParams object
                    params = VideoParams(
                        video_subject=video_subject,
                        video_script=video_script,
                        video_terms=video_terms.split(",") if video_terms else [],
                        video_concat_mode=VideoConcatMode(video_concat_mode[0]),
                        video_aspect=VideoAspect(video_aspect[0]),
                        video_transition_mode=VideoTransitionMode(video_transition[0]) if video_transition[0] else None,
                        video_clip_duration=clip_duration,
                        video_count=video_count,
                        voice_name=voice_name,
                        voice_rate=voice_rate,
                        voice_volume=voice_volume,
                        bgm_type=bgm_type[0],
                        bgm_file=bgm_file if bgm_type[0] == "custom" else None,
                        bgm_volume=bgm_volume,
                        subtitle_enabled=subtitle_enabled,
                        subtitle_font=subtitle_font if subtitle_enabled else None,
                        subtitle_position=subtitle_position[0] if subtitle_enabled else None,
                        subtitle_font_size=subtitle_font_size if subtitle_enabled else None,
                        subtitle_font_color=subtitle_font_color if subtitle_enabled else None,
                        subtitle_stroke_color=subtitle_stroke_color if subtitle_enabled else None,
                        subtitle_stroke_width=subtitle_stroke_width if subtitle_enabled else None,
                        video_source="pexels",  # Default source
                        video_language=script_language.lower() if script_language != "Auto Detect" else "auto"
                    )
                    
                    # Add task to session state
                    task_info = {
                        "id": task_id,
                        "status": "running",
                        "created_at": datetime.now().isoformat(),
                        "params": params.__dict__
                    }
                    st.session_state["tasks"].append(task_info)
                    
                    # Update progress
                    progress_placeholder.progress(0.1)
                    status_placeholder.info("🎬 Starting video generation...")
                    
                    # Call the main generation function
                    # This would typically be done asynchronously in a real app
                    result = tm.start(task_id, params, log_received)
                    
                    if result:
                        progress_placeholder.progress(1.0)
                        status_placeholder.success(locale_text.get("Video Generation Completed", "✅ Video generation completed!"))
                        
                        # Update task status
                        for task in st.session_state["tasks"]:
                            if task["id"] == task_id:
                                task["status"] = "completed"
                                break
                        
                        # Display download links
                        st.success(locale_text.get("You can download the generated video from the following links", 
                                                 "Download your generated video:"))
                        
                        # Show video preview if available
                        import os
                        from app.utils import utils
                        video_path = os.path.join(utils.task_dir(task_id), "combined.mp4")
                        if os.path.exists(video_path):
                            st.video(video_path)
                            
                            # Download button
                            with open(video_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Video",
                                    data=f.read(),
                                    file_name=f"video_{task_id[:8]}.mp4",
                                    mime="video/mp4"
                                )
                    else:
                        progress_placeholder.progress(0.0)
                        status_placeholder.error(locale_text.get("Video Generation Failed", "❌ Video generation failed"))
                        
                        # Update task status
                        for task in st.session_state["tasks"]:
                            if task["id"] == task_id:
                                task["status"] = "failed"
                                break
                        
                except Exception as e:
                    progress_placeholder.progress(0.0)
                    status_placeholder.error(f"❌ Error: {str(e)}")
                    logger.error(f"Video generation error: {str(e)}")

# Advanced Features Section
st.markdown("---")
st.subheader("🔧 Advanced Features")

# Template Management
with st.expander("📁 Template Management"):
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.write("**Save Current Settings as Template**")
        template_name = st.text_input("Template Name", key="new_template_name")
        if st.button("💾 Save Template", key='save_template_main'):
            if template_name:
                template_data = {
                    "video_subject": st.session_state.get("video_subject", ""),
                    "video_script": st.session_state.get("video_script", ""),
                    "video_terms": st.session_state.get("video_terms", ""),
                    "video_concat_mode": video_concat_mode,
                    "video_aspect": video_aspect,
                    "voice_name": voice_name,
                    "subtitle_enabled": subtitle_enabled
                }
                st.session_state["templates"][template_name] = template_data
                st.success(f"✅ Template '{template_name}' saved!")
            else:
                st.warning("Please enter a template name")

    with col_t2:
        st.write("**Load Template**")
        if st.session_state.get("templates"):
            selected_template = st.selectbox(
                "Select Template",
                options=list(st.session_state["templates"].keys()),
                key="template_selector"
            )
            if st.button("📂 Load Template"):
                template_data = st.session_state["templates"][selected_template]
                for key, value in template_data.items():
                    st.session_state[key] = value
                st.success(f"✅ Template '{selected_template}' loaded!")
                st.rerun()
        else:
            st.info("No templates saved yet")

# Batch Processing
with st.expander("🔄 Batch Processing"):
    st.write("**Batch Video Generation**")
    batch_subjects = st.text_area(
        "Enter multiple video subjects (one per line)",
        height=100,
        placeholder="Subject 1\nSubject 2\nSubject 3"
    )
    
    if st.button("🚀 Start Batch Generation"):
        if batch_subjects:
            subjects = [s.strip() for s in batch_subjects.split("\n") if s.strip()]
            st.info(f"Starting batch generation for {len(subjects)} videos...")
            # Batch processing would be implemented here
        else:
            st.warning("Please enter at least one subject")

# System Status
with st.expander("📊 System Status"):
    col_sys1, col_sys2 = st.columns(2)
    
    with col_sys1:
        st.metric("Active Tasks", len([t for t in st.session_state.get("tasks", []) if t.get("status") == "running"]))
        st.metric("Completed Tasks", len([t for t in st.session_state.get("tasks", []) if t.get("status") == "completed"]))
    
    with col_sys2:
        st.metric("Failed Tasks", len([t for t in st.session_state.get("tasks", []) if t.get("status") == "failed"]))
        st.metric("Total Templates", len(st.session_state.get("templates", {})))

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        MoneyPrinterTurbo Enhanced | Version 2.0 | 
        <a href='https://github.com/harry0703/MoneyPrinterTurbo' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

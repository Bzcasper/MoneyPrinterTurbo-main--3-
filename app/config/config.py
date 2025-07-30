import os
import shutil
import socket

import toml
from loguru import logger

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
config_file = f"{root_dir}/config.toml"


def load_config():
    # fix: IsADirectoryError: [Errno 21] Is a directory: '/MoneyPrinterTurbo/config.toml'
    if os.path.isdir(config_file):
        shutil.rmtree(config_file)

    if not os.path.isfile(config_file):
        example_file = f"{root_dir}/config.example.toml"
        if os.path.isfile(example_file):
            shutil.copyfile(example_file, config_file)
            logger.info("copy config.example.toml to config.toml")

    logger.info(f"load config from file: {config_file}")

    try:
        _config_ = toml.load(config_file)
    except Exception as e:
        logger.warning(f"load config failed: {str(e)}, try to load as utf-8-sig")
        with open(config_file, mode="r", encoding="utf-8-sig") as fp:
            _cfg_content = fp.read()
            _config_ = toml.loads(_cfg_content)
    return _config_


def save_config():
    with open(config_file, "w", encoding="utf-8") as f:
        _cfg["app"] = app
        _cfg["azure"] = azure
        _cfg["siliconflow"] = siliconflow
        _cfg["ui"] = ui
        f.write(toml.dumps(_cfg))


_cfg = load_config()
app = _cfg.get("app", {})

# CRITICAL FIX: Merge all app-level configs from different sections
# Some configs are under [app] section, others are at root level in [app]
# This ensures all LLM provider configs are available in config.app
if "llm_provider" not in app and "llm_provider" in _cfg.get("app", {}):
    # Update app dict with all app-level configs from the full config
    app.update(_cfg.get("app", {}))

# Also merge any root-level app configs that might be missing
root_app_configs = {k: v for k, v in _cfg.items() if k.startswith(('llm_', 'openai_', 'gemini_', 'azure_', 'moonshot_', 'qwen_', 'deepseek_', 'g4f_', 'oneapi_', 'cloudflare_', 'ernie_', 'pollinations_', 'ollama_'))}
app.update(root_app_configs)
whisper = _cfg.get("whisper", {})
proxy = _cfg.get("proxy", {})
azure = _cfg.get("azure", {})
siliconflow = _cfg.get("siliconflow", {})
ui = _cfg.get(
    "ui",
    {
        "hide_log": False,
    },
)
hive_memory = _cfg.get(
    "hive_memory",
    {
        "db_type": "sqlite",
        "db_path": "./memory/hive_memory.db",
        "max_connections": 10,
        "auto_create_tables": True,
        "retention_days": 30
    },
)

hostname = socket.gethostname()

log_level = _cfg.get("log_level", "DEBUG")
listen_host = _cfg.get("listen_host", "0.0.0.0")
listen_port = _cfg.get("listen_port", 8080)
project_name = _cfg.get("project_name", "MoneyPrinterTurbo")
project_description = _cfg.get(
    "project_description",
    "<a href='https://github.com/harry0703/MoneyPrinterTurbo'>https://github.com/harry0703/MoneyPrinterTurbo</a>",
)
project_version = _cfg.get("project_version", "1.2.6")
reload_debug = False

imagemagick_path = app.get("imagemagick_path", "")
if imagemagick_path and os.path.isfile(imagemagick_path):
    os.environ["IMAGEMAGICK_BINARY"] = imagemagick_path

ffmpeg_path = app.get("ffmpeg_path", "")
if ffmpeg_path and os.path.isfile(ffmpeg_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

logger.info(f"{project_name} v{project_version}")

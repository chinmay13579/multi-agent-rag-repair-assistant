import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models
TEXT_MODEL   = "nvidia/nemotron-3-super-120b-a12b:free"
VISION_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

# PDF
PDF_PATH = os.getenv("PDF_PATH", "../PE_Case_Study_Repair_Manual_Assistant.pdf")

# Retrieval
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
TOP_K_CHUNKS  = 4

# Agent
MAX_AGENT_ITERATIONS = 20

# Image rendering
IMAGE_DPI_SCALE = 2.0

# Paths
IMAGES_DIR        = "images"
CACHE_DIR         = "cache"
CHUNKS_CACHE      = "cache/chunks.json"
IMAGE_INDEX_CACHE = "cache/image_index.json"

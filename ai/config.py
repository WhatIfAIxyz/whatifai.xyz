"""
📌 HypotheticaNet - AI Model Configuration

This file contains all the configurable parameters for training, evaluation, and blockchain interactions.
"""

# ========================
# 🔥 AI Model Configuration
# ========================
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Using LLaMA-2 for better contextual reasoning
TOKENIZER_NAME = MODEL_NAME  # Use the same tokenizer as the model

# Training Hyperparameters
LEARNING_RATE = 2e-5  # Fine-tuning rate for optimal performance
BATCH_SIZE = 32  # Number of samples per batch
EPOCHS = 5  # Number of training cycles
GRADIENT_ACCUMULATION_STEPS = 4  # For effective batch size

# Hardware Settings
DEVICE = "cuda"  # Use "cuda" for GPU acceleration or "cpu" for CPU-only mode
MIXED_PRECISION = True  # Enables AMP (Automatic Mixed Precision) for faster training

# ========================
# 📂 Data Paths
# ========================
DATASET_NAME = "hypothetica/historical-counterfactuals"  # Custom dataset for training
DATA_PATH = "./data/"  # Base directory for datasets
TRAIN_DATA_PATH = f"{DATA_PATH}train.jsonl"  # Training dataset
TEST_DATA_PATH = f"{DATA_PATH}test.jsonl"  # Evaluation dataset

# Output & Model Saving
MODEL_SAVE_PATH = "./models/hypothetica_net.pth"  # Trained model storage
LOGS_PATH = "./logs/training_logs.json"  # Logs for tracking performance

# ========================
# ⛓️ Blockchain Configuration
# ========================
# Solana RPC Settings for logging AI-generated responses to blockchain
SOLANA_RPC_URL = "https://solana-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_API_KEY"
SOLANA_PRIVATE_KEY = "your_private_key_here"  # 🔒 Store securely (use environment variables)
SOLANA_WALLET_ADDRESS = "your_wallet_address_here"  # AI-generated data ownership

# Blockchain Smart Contract for AI Response Verification
SMART_CONTRACT_ADDRESS = "9zoP3nThLz8U3A7QfjgczNnQjGr1aDgK9yo9jHQa4cMb"

# ========================
# 🔗 API Configuration
# ========================
# API settings for serving AI model as a service
API_HOST = "0.0.0.0"
API_PORT = 8000
API_VERSION = "v1"
ENABLE_CORS = True  # Allows external applications to call the API

# ========================
# 📡 Logging & Debugging
# ========================
ENABLE_DEBUG = False  # Set to True for verbose debugging
LOGGING_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
MAX_LOG_FILE_SIZE_MB = 5  # Rotate logs when exceeding this limit

# ========================
# 🛠 Feature Flags
# ========================
ENABLE_BLOCKCHAIN_LOGGING = True  # Stores AI responses on-chain
ENABLE_ADVANCED_MEMORY = True  # Enables long-term context retention
USE_TENSOR_PARALLELISM = False  # Distribute model computation across multiple GPUs

# ========================
# 📝 Notes
# ========================
# - Update `SOLANA_PRIVATE_KEY` before deploying blockchain integration
# - Modify `MODEL_NAME` if switching to a different AI model
# - Use `.env` or secret managers for storing API keys securely

print(f"✅ HypotheticaNet Configuration Loaded | Model: {MODEL_NAME} | Device: {DEVICE}")

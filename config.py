# モデルに関する設定
MODEL_CONFIG = {
    "model_path": "Z:\models\GGUF\Mistral-Nemo-Japanese-Instruct-2408.Q4_K_M.gguf",
    "temperature": 0,
    "max_new_tokens": 1024 * 5,
    "context_window": 1024 * 10,
    "model_kwargs": {"n_gpu_layers": -1},
    "embedding_device": "cpu",
    "verbose": False,
    "stop_words": [
        "user:",
        "assistant:",
        "user\n",
        "assistant\n"
    ],
    "cache_folder": "./sentence_transformers",
    "embedding_model_path": "Z:\models\multilingual-e5-large",
    "index_mode": "DocumentSummaryIndex"
}

CHUNK_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 20,
    "delimiter": "-----",
    "data_directory": "data",
    "persist_dir": "index",
    "retrieval_top_k": 5
}
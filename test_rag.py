import sys
from rag import ExperimentStore
import config
from pathlib import Path

store = ExperimentStore(
    embedding_model_name=config.get_config().rag.embedding_model,
    embedding_dim=config.get_config().rag.embedding_dim,
    persist_dir=config.get_config().paths.rag_dir,
)
try:
    store.load(config.get_config().paths.rag_dir)
    docs = store.query("test", top_k=3)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()

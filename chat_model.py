import os
import logging
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    DocumentSummaryIndex,
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from config import MODEL_CONFIG, CHUNK_CONFIG


class ChatModel:
    """
    LLM、埋め込みモデル、ServiceContext を用いて
    DocumentSummaryIndex の作成／ロードを行うクラス
    """
    def __init__(self):
        self.llm = LlamaCPP(
            model_path=MODEL_CONFIG["model_path"],
            temperature=MODEL_CONFIG["temperature"],
            max_new_tokens=MODEL_CONFIG["max_new_tokens"],
            context_window=MODEL_CONFIG["context_window"],
            generate_kwargs={},
            model_kwargs=MODEL_CONFIG["model_kwargs"],
            verbose=MODEL_CONFIG["verbose"],
        )
        self.embed_model = HuggingFaceEmbedding(
            model_name=MODEL_CONFIG["embedding_model_path"],
            cache_folder=MODEL_CONFIG["cache_folder"],
            device=MODEL_CONFIG["embedding_device"]
        )
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        self.index = self.get_index()

    def get_index(self, persist_dir=CHUNK_CONFIG["persist_dir"]):
        """
        persist_dir が存在していれば保存済みのインデックスをロードする
        存在しない場合または読み込みに失敗した場合、新規作成して永続化する
        """
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            logging.info("既存の DocumentSummaryIndex をロードしました。")
            return index
        
        except:
            return self.create_index()

    def create_index(
            self,
            persist_dir=CHUNK_CONFIG["persist_dir"],
            delimiter=CHUNK_CONFIG["delimiter"],
            data_directory=CHUNK_CONFIG["data_directory"]
        ):
        """
        インデックスを新規作成して永続化する
        """
        documents = self._split_documents_by_delimiter(data_directory, delimiter)
        if MODEL_CONFIG["index_mode"] == "DocumentSummaryIndex":
            index = DocumentSummaryIndex.from_documents(documents)
        elif MODEL_CONFIG["index_mode"] == "VectorStoreIndex":
            index = VectorStoreIndex.from_documents(documents)
        else:
            raise NotImplementedError(f"Undefined index mode: {MODEL_CONFIG['index_mode']}")
    
        index.storage_context.persist(persist_dir=persist_dir)
        logging.info("新規 DocumentSummaryIndex を作成・保存しました。")
        return index

    def _split_documents_by_delimiter(self, directory, delimiter):
        documents = []
        reader = SimpleDirectoryReader(directory)
        raw_docs = reader.load_data()
        for doc in raw_docs:
            chunks = doc.text.split(delimiter)
            for chunk in chunks:
                if chunk.strip():
                    documents.append(Document(text=chunk.strip()))
        return documents
from typing import List, Optional
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str = Field(..., description="送信者のロール（例: 'user' または 'assistant'）")
    content: str = Field(..., description="メッセージの内容")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(
        ...,
        description="会話のメッセージリスト。リストの最後がユーザークエリとして扱われます。"
    )
    system_prompt_override: Optional[str] = Field(default="", description="上書きするシステムプロンプト")
    use_index: Optional[bool] = Field(
        default=True,
        description="インデックスを使ったRAG機能を利用するかどうか"
    )
    retrieval_top_k: Optional[int] = Field(
        default=8,
        description="RAG検索で取得する件数"
    )
    retrieval_role: Optional[str] = Field(default="", description="検索して得られた情報の役割")

class ChatChoice(BaseModel):
    message: Message = Field(..., description="生成されたアシスタントのメッセージ")

class ChatResponse(BaseModel):
    id: str = Field(..., description="チャット会話の一意な識別子")
    object: str = Field(..., description="レスポンスのタイプ（例: 'chat.completion'）")
    choices: List[ChatChoice] = Field(
        ...,
        description="生成された選択肢（複数候補があればリストで返す）"
    )

class ErrorResponse(BaseModel):
    error: str = Field(..., description="エラー内容の説明")
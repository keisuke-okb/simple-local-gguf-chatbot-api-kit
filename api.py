# api.py

import asyncio
import logging
from fastapi import FastAPI, Request, HTTPException, Path as FastAPIPath
from fastapi.responses import StreamingResponse, FileResponse
from pathlib import Path

from chat_model import ChatModel
from config import CHUNK_CONFIG, MODEL_CONFIG
from prompt_config import SYSTEM_PROMPT, RETRIEVAL_ROLE
from prompt import build_chat_messages
from api_models import Message, ChatRequest, ChatChoice, ChatResponse, ErrorResponse

app = FastAPI(
    title="Simple Local Chatbot API Kit",
    description=(
        "このAPIは、VectorStoreIndex/DocumentSummaryIndexを用いたシンプルなRAG機能を持つチャットボットAPIを実現します。\n"
        "リクエストにはOpenAI Chat Completion APIフォーマットに準拠した会話履歴を含める必要があります。"
    ),
    version="1.0.0"
)

# モデルの初期化
chat_model = ChatModel()
index = chat_model.index
retrieval_top_k = CHUNK_CONFIG["retrieval_top_k"]

# バッチ処理エンドポイント（非ストリーミング）
@app.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error: 回答生成に失敗しました。"}
    },
    summary="チャット応答（バッチ処理）",
    description=(
        "リクエストに含まれる会話履歴から、関連文書の取得（RAG）と LlamaCPP の stream_chat() "
        "を用いた推論を実現します。"
    )
)
async def chat_batch(request: Request, chat_request: ChatRequest):
    """
    【リクエスト例】
    {
        "messages": [
            { "role": "system", "content": "あなたは親切なAIアシスタントです。" },
            { "role": "user", "content": "こんにちは" },
            { "role": "assistant", "content": "こんにちは。本日はどうされましたか？" },
            { "role": "user", "content": "最新の天気を教えてほしいです。" }
        ]
    }
    
    【レスポンス例】
    {
        "id": "chat-1",
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "今日の天気は晴れです。"
                }
            }
        ]
    }
    """
    try:
        if not chat_request.messages or len(chat_request.messages) < 1:
            raise HTTPException(status_code=400, detail="メッセージリストが空です。")
        
        conversation_history = chat_request.messages[:-1]
        user_message = chat_request.messages[-1].content

        if chat_request.use_index:
            retriever = index.as_retriever(similarity_top_k=chat_request.retrieval_top_k)
            retrieved_docs = retriever.retrieve(user_message)
            retrieved_texts = [doc.text for doc in retrieved_docs]
        else:
            retrieved_texts = None

        system_prompt = chat_request.system_prompt_override \
            if chat_request.system_prompt_override != "" else SYSTEM_PROMPT
        
        retrieval_role = chat_request.retrieval_role \
            if chat_request.retrieval_role != "" else RETRIEVAL_ROLE

        messages = build_chat_messages(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            retrieval_role=retrieval_role,
            retrieved_texts=retrieved_texts,
            user_query=user_message
        )

        # LlamaCPP の stream_chat() を用いて推論し、すべてのトークンを連結
        response_text = ""
        for token in chat_model.llm.stream_chat(messages):
            if any([stop_word in response_text + token.delta for stop_word in MODEL_CONFIG["stop_words"]]):
                print("[STOP WORD DETECTED]")
                break
            response_text += token.delta

        return ChatResponse(
            id="chat-1",
            object="chat.completion",
            choices=[ChatChoice(message=Message(role="assistant", content=response_text))]
        )
    
    except Exception as e:
        logging.exception("chat_batch エンドポイントでエラーが発生しました")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ストリーミング処理エンドポイント
@app.post(
    "/v1/chat/completions/streaming",
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error: ストリーミング生成に失敗しました。"}
    },
    summary="チャット応答（ストリーミング処理）",
    description=(
        "リクエストに含まれる会話履歴から、関連文書の取得（RAG）と LlamaCPP の stream_chat() "
        "を用いた推論を実現し、トークンを1件ずつストリーミング形式で返却します。"
    )
)
async def chat_streaming(request: Request, chat_request: ChatRequest):
    """
    ストリーミングレスポンスでは、生成された回答が 1 トークンずつ送信され、最終的に [DONE] を送信します。
    
    【リクエスト例】
    {
        "messages": [
            { "role": "system", "content": "あなたは親切なAIアシスタントです。" },
            { "role": "user", "content": "こんにちは" },
            { "role": "assistant", "content": "こんにちは。本日はどうされましたか？" },
            { "role": "user", "content": "最新の天気を教えてほしいです。" }
        ]
    }
    """
    try:
        if not chat_request.messages or len(chat_request.messages) < 1:
            raise HTTPException(status_code=400, detail="メッセージリストが空です。")
        
        conversation_history = chat_request.messages[:-1]
        user_message = chat_request.messages[-1].content

        if chat_request.use_index:
            retriever = index.as_retriever(similarity_top_k=chat_request.retrieval_top_k)
            retrieved_docs = retriever.retrieve(user_message)
            retrieved_texts = [doc.text for doc in retrieved_docs]
        else:
            retrieved_texts = None

        system_prompt = chat_request.system_prompt_override \
            if chat_request.system_prompt_override != "" else SYSTEM_PROMPT
        
        retrieval_role = chat_request.retrieval_role \
            if chat_request.retrieval_role != "" else RETRIEVAL_ROLE

        messages = build_chat_messages(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            retrieval_role=retrieval_role,
            retrieved_texts=retrieved_texts,
            user_query=user_message
        )

        print(messages)
        response = chat_model.llm.stream_chat(messages)

        async def event_generator():
            response_text = ""
            for token in response:
                if any([stop_word in response_text + token.delta for stop_word in MODEL_CONFIG["stop_words"]]):
                    print("[STOP WORD DETECTED]")
                    break
                print(token.delta, end="", flush=True)
                escaped_token = token.delta.replace("\n", "\\n")
                response_text += token.delta
                yield f"data: {escaped_token}\n\n"
                await asyncio.sleep(0.01)

            # すべてのトークン送信後、[DONE] を送信
            print()
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    except Exception as e:
        logging.exception("chat_streaming エンドポイントでエラーが発生しました")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    

# フロントエンド用Webアプリ
BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"

@app.get(
    "/ui/{file_path:path}",
    summary="UI静的コンテンツの取得",
    description="""
    このエンドポイントは、`ui` フォルダ内の静的ファイルを返却します。  
    パスパラメータ `file_path` には、`ui` フォルダ内の対象ファイルの相対パスを指定してください。

    - 指定されたファイルが存在し、`ui` フォルダ内にある場合は、そのファイルをそのまま返します。
    - セキュリティ上の理由から、`../` などを使用して `ui` フォルダ外へのアクセスは拒否され、存在しないファイルも 404 エラーが返されます。
    """
)
async def serve_ui_file(
    file_path: str = FastAPIPath(..., description="返却対象のファイルの相対パス。例: 'images/logo.png'")
):
    try:
        # リクエストされたファイルの絶対パスを解決
        requested_file = (UI_DIR / file_path).resolve()

        # セキュリティチェック: リクエストされたファイルが ui フォルダ内にあるか確認
        if UI_DIR not in requested_file.parents:
            raise HTTPException(status_code=404, detail="File not found")

        # ファイルが存在し、通常のファイルであれば返す
        if requested_file.exists() and requested_file.is_file():
            return FileResponse(str(requested_file))
        else:
            raise HTTPException(status_code=404, detail="File not found")
        
    except:
        raise HTTPException(status_code=400, detail="Invalid request")
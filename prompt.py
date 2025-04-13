from llama_index.core.llms import ChatMessage

def build_chat_messages(system_prompt, conversation_history, retrieval_role, retrieved_texts, user_query):
    """
    システムプロンプト、会話履歴、取得された関連文書、ユーザークエリから
    LlamaCPP 用のメッセージリスト（OpenAI Chat API形式）を生成する。
    
    引数:
        - system_prompt (str): システムプロンプト
        - conversation_history (list): 過去の会話履歴。各要素は api_models.py の Message オブジェクト
        - retrieved_texts (list): 取得された関連文書のテキストの一覧
        - user_query (str): 最新のユーザークエリ
        
    戻り値:
        - messages (list): {"role": str, "content": str} の形式のリスト
    """
    messages = []
    messages.append(ChatMessage(role="system", content=system_prompt))
    # 会話履歴の追加（システム以外の役割をそのまま反映）
    for msg in conversation_history:
        messages.append(ChatMessage(role=msg.role, content=msg.content))
    
    # 取得された関連文書をひとまとめのシステムメッセージとして追加
    if retrieved_texts:
        context = "\n\n----------\n\n".join(retrieved_texts)
        messages.append(ChatMessage(role="system", content=f"{retrieval_role}：\n{context}"))
    
    # 最新のユーザークエリを追加
    messages.append(ChatMessage(role="user", content=user_query))
    return messages
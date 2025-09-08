# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI Agent Framework: MCPツールコールLangGraphエージェントの作成とデプロイ
# MAGIC
# MAGIC このノートブックでは、Databricks上でホストされているMCPサーバーに接続するLangGraphエージェントの作成方法を説明します。Databricks管理のMCPサーバー、DatabricksアプリとしてホストされたカスタムMCPサーバー、または両方を同時に選択できます。詳細は[MCP on Databricks](https://docs.databricks.com/aws/ja/generative-ai/mcp/)をご覧ください。
# MAGIC
# MAGIC このノートブックでは、Mosaic AIの機能と互換性のある[`ResponsesAgent`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ResponsesAgent)インターフェースを使用します。このノートブックで学べること：
# MAGIC
# MAGIC - MCPツールを呼び出すLangGraphエージェント（`ResponsesAgent`でラップ）を作成
# MAGIC - エージェントの手動テスト
# MAGIC - Mosaic AI Agent Evaluationによるエージェントの評価
# MAGIC - エージェントのログ化とデプロイ
# MAGIC
# MAGIC Mosaic AI Agent Frameworkを使ったエージェント作成の詳細は、Databricksドキュメント（[AWS](https://docs.databricks.com/aws/ja/generative-ai/agent-framework/author-agent) | [Azure](https://learn.microsoft.com/ja-jp/azure/databricks/generative-ai/agent-framework/author-agent)）をご覧ください。
# MAGIC
# MAGIC ## 前提条件
# MAGIC
# MAGIC - このノートブック内のすべての`TODO`を対応してください。

# COMMAND ----------

# MAGIC %pip install -U -qqqq mcp>=1.9 databricks-sdk[openai] databricks-agents>=1.0.0 databricks-mcp databricks-langchain uv langgraph==0.3.4
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### エージェントコードの定義
# MAGIC
# MAGIC 以下のセルでエージェントコードを1つのセルにまとめて定義します。これにより、`%%writefile`マジックコマンドを使ってローカルのPythonファイルに簡単に書き出し、その後のログ化やデプロイに利用できます。
# MAGIC
# MAGIC このセルでは、Databricks MCPサーバーとMosaic AI Agent Frameworkを統合した柔軟なツール利用エージェントを作成します。概要は以下の通りです：
# MAGIC
# MAGIC 1. **MCPツールラッパー**  
# MAGIC    LangChainツールがDatabricks MCPサーバーとやり取りできるように、カスタムラッパークラスを定義します。Databricks管理MCPサーバー、カスタムMCPサーバー、または両方に接続可能です。
# MAGIC
# MAGIC 2. **ツールの発見と登録**  
# MAGIC    指定したMCPサーバーから利用可能なツールを自動的に検出し、そのスキーマをPythonオブジェクトに変換し、LLM用に準備します。
# MAGIC
# MAGIC 3. **LangGraphエージェントロジックの定義**  
# MAGIC    エージェントのワークフローを定義します：
# MAGIC    - エージェントがメッセージ（会話/履歴）を読み取る
# MAGIC    - ツール（関数）コールが要求された場合、正しいMCPツールを使って実行
# MAGIC    - 必要に応じて複数回ツールコールを繰り返し、最終回答が用意できるまでループ
# MAGIC
# MAGIC 4. **LangGraphエージェントを`ResponsesAgent`クラスでラップ**  
# MAGIC    エージェントを`ResponsesAgent`でラップし、Mosaic AIと互換性を持たせます。
# MAGIC    
# MAGIC 5. **MLflow自動トレース**
# MAGIC    MLflowの自動ロギングを有効化し、自動トレースを開始します。

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import asyncio
# MAGIC import mlflow
# MAGIC import os
# MAGIC import json
# MAGIC from uuid import uuid4
# MAGIC from pydantic import BaseModel, create_model
# MAGIC from typing import Annotated, Any, Generator, List, Optional, Sequence, TypedDict, Union
# MAGIC
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     UCFunctionToolkit,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC )
# MAGIC from databricks_mcp import DatabricksOAuthClientProvider, DatabricksMCPClient
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.messages import (
# MAGIC     AIMessage,
# MAGIC     AIMessageChunk,
# MAGIC     BaseMessage,
# MAGIC     convert_to_openai_messages,
# MAGIC )
# MAGIC from langchain_core.tools import BaseTool, tool
# MAGIC
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC
# MAGIC from mcp import ClientSession
# MAGIC from mcp.client.streamable_http import streamablehttp_client as connect
# MAGIC
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC )
# MAGIC
# MAGIC import nest_asyncio
# MAGIC nest_asyncio.apply()
# MAGIC
# MAGIC ############################################
# MAGIC ## LLMエンドポイントとシステムプロンプトを定義
# MAGIC ############################################
# MAGIC # TODO: モデルサービングエンドポイント名を指定してください
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC # TODO: システムプロンプトを編集してください
# MAGIC system_prompt = "あなたはPythonコードを実行できる有能なアシスタントです。"
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## エージェント用MCPサーバーの設定
# MAGIC ## このセクションでは、エージェントがデータ取得やアクション実行のためにサーバー接続を設定します。
# MAGIC ###############################################################################
# MAGIC
# MAGIC # TODO: MCPサーバー接続タイプを選択してください。
# MAGIC
# MAGIC # ----- シンプル: 管理MCPサーバー（追加設定不要） -----
# MAGIC # Databricksワークスペース設定とPAT認証を利用します。
# MAGIC workspace_client = WorkspaceClient()
# MAGIC
# MAGIC # 管理MCPサーバー: 上記のデフォルト設定で利用可能
# MAGIC host = workspace_client.config.host
# MAGIC MANAGED_MCP_SERVER_URLS = [
# MAGIC     f"{host}/api/2.0/mcp/functions/system/ai",
# MAGIC ]
# MAGIC
# MAGIC # ----- 上級者向け（オプション）: OAuthを使ったカスタムMCPサーバー -----
# MAGIC # DatabricksアプリでホストされるカスタムMCPサーバーには、サービスプリンシパルによるOAuthが必要です。
# MAGIC # 下記はカスタムMCPサーバー接続時のみコメントアウトを外して設定してください。
# MAGIC #
# MAGIC # import os
# MAGIC # workspace_client = WorkspaceClient(
# MAGIC #     host="<DATABRICKS_WORKSPACE_URL>",
# MAGIC #     client_id=os.getenv("DATABRICKS_CLIENT_ID"),
# MAGIC #     client_secret=os.getenv("DATABRICKS_CLIENT_SECRET"),
# MAGIC #     auth_type="oauth-m2m",   # マシン間OAuthを有効化
# MAGIC # )
# MAGIC
# MAGIC # カスタムMCPサーバー: 必要に応じて下記にURLを追加（上記OAuth設定が必要）
# MAGIC CUSTOM_MCP_SERVER_URLS = [
# MAGIC     # 例: "https://<custom-mcp-url>/mcp"
# MAGIC ]
# MAGIC
# MAGIC #####################
# MAGIC ## MCPツールの作成
# MAGIC #####################
# MAGIC
# MAGIC # MCPサーバー呼び出し機能をラップするカスタムLangChainツールを定義
# MAGIC class MCPTool(BaseTool):
# MAGIC     """MCPサーバー機能をラップするカスタムLangChainツール"""
# MAGIC
# MAGIC     def __init__(self, name: str, description: str, args_schema: type, server_url: str, ws: WorkspaceClient, is_custom: bool = False):
# MAGIC         # ツールを初期化
# MAGIC         super().__init__(
# MAGIC             name=name,
# MAGIC             description=description,
# MAGIC             args_schema=args_schema
# MAGIC         )
# MAGIC         # MCPサーバーURL、Databricksワークスペースクライアント、カスタムサーバー用フラグを属性として保存
# MAGIC         object.__setattr__(self, 'server_url', server_url)
# MAGIC         object.__setattr__(self, 'workspace_client', ws)
# MAGIC         object.__setattr__(self, 'is_custom', is_custom)
# MAGIC
# MAGIC     def _run(self, **kwargs) -> str:
# MAGIC         """MCPツールを実行"""
# MAGIC         if self.is_custom:
# MAGIC             # カスタムMCPサーバー用（OAuth必須）は非同期メソッドを利用
# MAGIC             return asyncio.run(self._run_custom_async(**kwargs))
# MAGIC         else:
# MAGIC             # 管理MCPサーバーは同期呼び出し
# MAGIC             mcp_client = DatabricksMCPClient(server_url=self.server_url, workspace_client=self.workspace_client)
# MAGIC             response = mcp_client.call_tool(self.name, kwargs)
# MAGIC             return "".join([c.text for c in response.content])
# MAGIC
# MAGIC     async def _run_custom_async(self, **kwargs) -> str:
# MAGIC         """カスタムMCPツールを非同期で実行"""        
# MAGIC         async with connect(self.server_url, auth=DatabricksOAuthClientProvider(self.workspace_client)) as (
# MAGIC             read_stream,
# MAGIC             write_stream,
# MAGIC             _,
# MAGIC         ):
# MAGIC             # サーバーとの非同期セッションを作成しツールを呼び出す
# MAGIC             async with ClientSession(read_stream, write_stream) as session:
# MAGIC                 await session.initialize()
# MAGIC                 response = await session.call_tool(self.name, kwargs)
# MAGIC                 return "".join([c.text for c in response.content])
# MAGIC
# MAGIC # カスタムMCPサーバー（OAuth必須）からツール定義を取得
# MAGIC async def get_custom_mcp_tools(ws: WorkspaceClient, server_url: str):
# MAGIC     """OAuthを使ってカスタムMCPサーバーからツールを取得"""    
# MAGIC     async with connect(server_url, auth=DatabricksOAuthClientProvider(ws)) as (
# MAGIC         read_stream,
# MAGIC         write_stream,
# MAGIC         _,
# MAGIC     ):
# MAGIC         async with ClientSession(read_stream, write_stream) as session:
# MAGIC             await session.initialize()
# MAGIC             tools_response = await session.list_tools()
# MAGIC             return tools_response.tools
# MAGIC
# MAGIC # 管理MCPサーバーからツール定義を取得
# MAGIC def get_managed_mcp_tools(ws: WorkspaceClient, server_url: str):
# MAGIC     """管理MCPサーバーからツールを取得"""
# MAGIC     mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=ws)
# MAGIC     return mcp_client.list_tools()
# MAGIC
# MAGIC # MCPツール定義をLangChain互換ツールに変換
# MAGIC def create_langchain_tool_from_mcp(mcp_tool, server_url: str, ws: WorkspaceClient, is_custom: bool = False):
# MAGIC     """MCPツール定義からLangChainツールを作成"""
# MAGIC     schema = mcp_tool.inputSchema.copy()
# MAGIC     properties = schema.get("properties", {})
# MAGIC     required = schema.get("required", [])
# MAGIC
# MAGIC     # JSONスキーマ型をPython型にマッピング
# MAGIC     TYPE_MAPPING = {
# MAGIC         "integer": int,
# MAGIC         "number": float,
# MAGIC         "boolean": bool
# MAGIC     }
# MAGIC     field_definitions = {}
# MAGIC     for field_name, field_info in properties.items():
# MAGIC         field_type_str = field_info.get("type", "string")
# MAGIC         field_type = TYPE_MAPPING.get(field_type_str, str)
# MAGIC
# MAGIC         if field_name in required:
# MAGIC             field_definitions[field_name] = (field_type, ...)
# MAGIC         else:
# MAGIC             field_definitions[field_name] = (field_type, None)
# MAGIC
# MAGIC     # ツールの入力引数用にPydanticスキーマを動的生成
# MAGIC     args_schema = create_model(
# MAGIC         f"{mcp_tool.name}Args",
# MAGIC         **field_definitions
# MAGIC     )
# MAGIC
# MAGIC     # 設定済みMCPToolインスタンスを返す
# MAGIC     return MCPTool(
# MAGIC         name=mcp_tool.name,
# MAGIC         description=mcp_tool.description or f"Tool: {mcp_tool.name}",
# MAGIC         args_schema=args_schema,
# MAGIC         server_url=server_url,
# MAGIC         ws=ws,
# MAGIC         is_custom=is_custom
# MAGIC     )
# MAGIC
# MAGIC # 管理・カスタムMCPサーバーからすべてのツールをまとめて取得
# MAGIC async def create_mcp_tools(ws: WorkspaceClient, 
# MAGIC                           managed_server_urls: List[str] = None, 
# MAGIC                           custom_server_urls: List[str] = None) -> List[MCPTool]:
# MAGIC     """管理・カスタムMCPサーバーからLangChainツールを作成"""
# MAGIC     tools = []
# MAGIC
# MAGIC     if managed_server_urls:
# MAGIC         # 管理MCPツールをロード
# MAGIC         for server_url in managed_server_urls:
# MAGIC             try:
# MAGIC                 mcp_tools = get_managed_mcp_tools(ws, server_url)
# MAGIC                 for mcp_tool in mcp_tools:
# MAGIC                     tool = create_langchain_tool_from_mcp(mcp_tool, server_url, ws, is_custom=False)
# MAGIC                     tools.append(tool)
# MAGIC             except Exception as e:
# MAGIC                 print(f"管理サーバー{server_url}からツールのロードに失敗: {e}")
# MAGIC
# MAGIC     if custom_server_urls:
# MAGIC         # カスタムMCPツールを非同期でロード
# MAGIC         for server_url in custom_server_urls:
# MAGIC             try:
# MAGIC                 mcp_tools = await get_custom_mcp_tools(ws, server_url)
# MAGIC                 for mcp_tool in mcp_tools:
# MAGIC                     tool = create_langchain_tool_from_mcp(mcp_tool, server_url, ws, is_custom=True)
# MAGIC                     tools.append(tool)
# MAGIC             except Exception as e:
# MAGIC                 print(f"カスタムサーバー{server_url}からツールのロードに失敗: {e}")
# MAGIC
# MAGIC     return tools
# MAGIC
# MAGIC #####################
# MAGIC ## エージェントロジックの定義
# MAGIC #####################
# MAGIC
# MAGIC # エージェントワークフロー用の状態（会話やカスタムデータを含む）
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[BaseMessage], add_messages]
# MAGIC     custom_inputs: Optional[dict[str, Any]]
# MAGIC     custom_outputs: Optional[dict[str, Any]]
# MAGIC
# MAGIC # ツールコール可能なLangGraphエージェントを定義
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[ToolNode, Sequence[BaseTool]],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ):
# MAGIC     model = model.bind_tools(tools)  # モデルにツールをバインド
# MAGIC
# MAGIC     # 最後のメッセージに基づきエージェントが継続か終了かを判定
# MAGIC     def should_continue(state: AgentState):
# MAGIC         messages = state["messages"]
# MAGIC         last_message = messages[-1]
# MAGIC         # 関数（ツール）コールがあれば継続、なければ終了
# MAGIC         if isinstance(last_message, AIMessage) and last_message.tool_calls:
# MAGIC             return "continue"
# MAGIC         else:
# MAGIC             return "end"
# MAGIC
# MAGIC     # 前処理: 必要に応じてシステムプロンプトを会話履歴の先頭に追加
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC
# MAGIC     model_runnable = preprocessor | model  # 前処理とモデルをチェーン
# MAGIC
# MAGIC     # ワークフロー内でモデルを呼び出す関数
# MAGIC     def call_model(
# MAGIC         state: AgentState,
# MAGIC         config: RunnableConfig,
# MAGIC     ):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(AgentState)  # エージェントの状態マシンを作成
# MAGIC
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))  # エージェントノード（LLM）
# MAGIC     workflow.add_node("tools", ToolNode(tools))             # ツールノード
# MAGIC
# MAGIC     workflow.set_entry_point("agent")  # エージェントノードから開始
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {
# MAGIC             "continue": "tools",  # モデルがツールコールを要求したらツールノードへ
# MAGIC             "end": END,           # それ以外はワークフロー終了
# MAGIC         },
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")  # ツール実行後はエージェントノードに戻る
# MAGIC
# MAGIC     # ツールコールエージェントワークフローをコンパイルして返す
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC # Compiled agentをResponsesAgentでラップし、Mosaic AI Responses API互換に
# MAGIC class LangGraphResponsesAgent(ResponsesAgent):
# MAGIC     def __init__(self, agent):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     # Responses形式メッセージをChatCompletion形式に変換
# MAGIC     def _responses_to_cc(
# MAGIC         self, message: dict[str, Any]
# MAGIC     ) -> list[dict[str, Any]]:
# MAGIC         """Responses API出力アイテムをChatCompletionメッセージに変換"""
# MAGIC         msg_type = message.get("type")
# MAGIC         if msg_type == "function_call":
# MAGIC             # ツール/関数コールメッセージの整形
# MAGIC             return [
# MAGIC                 {
# MAGIC                     "role": "assistant",
# MAGIC                     "content": "tool call",
# MAGIC                     "tool_calls": [
# MAGIC                         {
# MAGIC                             "id": message["call_id"],
# MAGIC                             "type": "function",
# MAGIC                             "function": {
# MAGIC                                 "arguments": message["arguments"],
# MAGIC                                 "name": message["name"],
# MAGIC                             },
# MAGIC                         }
# MAGIC                     ],
# MAGIC                 }
# MAGIC             ]
# MAGIC         elif msg_type == "message" and isinstance(message["content"], list):
# MAGIC             # 通常のコンテンツメッセージの整形
# MAGIC             return [
# MAGIC                 {"role": message["role"], "content": content["text"]}
# MAGIC                 for content in message["content"]
# MAGIC             ]
# MAGIC         elif msg_type == "reasoning":
# MAGIC             # reasoningステップはassistantメッセージとして
# MAGIC             return [{"role": "assistant", "content": json.dumps(message["summary"])}]
# MAGIC         elif msg_type == "function_call_output":
# MAGIC             # 関数/ツールの出力
# MAGIC             return [
# MAGIC                 {
# MAGIC                     "role": "tool",
# MAGIC                     "content": message["output"],
# MAGIC                     "tool_call_id": message["call_id"],
# MAGIC                 }
# MAGIC             ]
# MAGIC         # 既知の互換フィールドのみ通す
# MAGIC         compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
# MAGIC         filtered = {k: v for k, v in message.items() if k in compatible_keys}
# MAGIC         return [filtered] if filtered else []
# MAGIC
# MAGIC     # LangChainメッセージをResponses形式辞書に変換
# MAGIC     def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
# MAGIC         """LangChainメッセージをResponses出力アイテム辞書に変換"""
# MAGIC         for message in messages:
# MAGIC             message = message.model_dump()  # モデルを辞書に変換
# MAGIC             role = message["type"]
# MAGIC             if role == "ai":
# MAGIC                 if tool_calls := message.get("tool_calls"):
# MAGIC                     # すべてのツールコールについてfunction callアイテムを返す
# MAGIC                     return [
# MAGIC                         self.create_function_call_item(
# MAGIC                             id=message.get("id") or str(uuid4()),
# MAGIC                             call_id=tool_call["id"],
# MAGIC                             name=tool_call["name"],
# MAGIC                             arguments=json.dumps(tool_call["args"]),
# MAGIC                         )
# MAGIC                         for tool_call in tool_calls
# MAGIC                     ]
# MAGIC                 else:
# MAGIC                     # 通常のAIテキストメッセージ
# MAGIC                     return [
# MAGIC                         self.create_text_output_item(
# MAGIC                             text=message["content"],
# MAGIC                             id=message.get("id") or str(uuid4()),
# MAGIC                         )
# MAGIC                     ]
# MAGIC             elif role == "tool":
# MAGIC                 # ツール/関数実行の出力
# MAGIC                 return [
# MAGIC                     self.create_function_call_output_item(
# MAGIC                         call_id=message["tool_call_id"],
# MAGIC                         output=message["content"],
# MAGIC                     )
# MAGIC                 ]
# MAGIC             elif role == "user":
# MAGIC                 # ユーザーメッセージはそのまま
# MAGIC                 return [message]
# MAGIC
# MAGIC     # エージェントの予測（単一ステップ）
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done" or event.type == "error"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     # エージェントのストリーム予測（逐次出力）
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         cc_msgs = []
# MAGIC         for msg in request.input:
# MAGIC             cc_msgs.extend(self._responses_to_cc(msg.model_dump()))
# MAGIC
# MAGIC         # エージェントグラフからイベントをストリーム
# MAGIC         for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
# MAGIC             if event[0] == "updates":
# MAGIC                 # ワークフローノードから更新メッセージをストリーム
# MAGIC                 for node_data in event[1].values():
# MAGIC                     if "messages" in node_data:
# MAGIC                         for item in self._langchain_to_responses(node_data["messages"]):
# MAGIC                             yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
# MAGIC             elif event[0] == "messages":
# MAGIC                 # 生成テキストメッセージチャンクをストリーム
# MAGIC                 try:
# MAGIC                     chunk = event[1][0]
# MAGIC                     if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
# MAGIC                         yield ResponsesAgentStreamEvent(
# MAGIC                             **self.create_text_delta(delta=content, item_id=chunk.id),
# MAGIC                         )
# MAGIC                 except:
# MAGIC                     pass
# MAGIC
# MAGIC # MCPツールとワークフローを含むエージェント全体を初期化
# MAGIC def initialize_agent():
# MAGIC     """MCPツール付きエージェントを初期化"""
# MAGIC     # 設定済みサーバーからMCPツールを作成
# MAGIC     mcp_tools = asyncio.run(create_mcp_tools(
# MAGIC         ws=workspace_client,
# MAGIC         managed_server_urls=MANAGED_MCP_SERVER_URLS,
# MAGIC         custom_server_urls=CUSTOM_MCP_SERVER_URLS
# MAGIC     ))
# MAGIC
# MAGIC     # LLM・ツールセット・システムプロンプト（必要に応じて）でエージェントグラフを作成
# MAGIC     agent = create_tool_calling_agent(llm, mcp_tools, system_prompt)
# MAGIC     return LangGraphResponsesAgent(agent)
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC AGENT = initialize_agent()
# MAGIC mlflow.models.set_model(AGENT)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## エージェントのテスト
# MAGIC
# MAGIC エージェントと対話して、その出力やツールコール機能をテストします。このノートブックでは`mlflow.langchain.autolog()`を呼び出しているため、エージェントの各ステップのトレースを確認できます。

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# ==============================================================================
# TODO: ONLY UNCOMMENT AND EDIT THIS SECTION IF YOU ARE USING OAUTH/SERVICE PRINCIPAL FOR CUSTOM MCP SERVERS.
#       For managed MCP (the default), LEAVE THIS SECTION COMMENTED OUT.
# ==============================================================================

# import os

# # Set your Databricks client ID and client secret for service principal authentication.
# DATABRICKS_CLIENT_ID = "<YOUR_CLIENT_ID>"
# client_secret_scope_name = "<YOUR_SECRET_SCOPE>"
# client_secret_key_name = "<YOUR_SECRET_KEY_NAME>"

# # Load your service principal credentials into environment variables
# os.environ["DATABRICKS_CLIENT_ID"] = DATABRICKS_CLIENT_ID
# os.environ["DATABRICKS_CLIENT_SECRET"] = dbutils.secrets.get(scope=client_secret_scope_name, key=client_secret_key_name)


# COMMAND ----------

from agent import AGENT

AGENT.predict({"input": [{"role": "user", "content": "Pythonで7*6は何ですか？"}]})

# COMMAND ----------

for chunk in AGENT.predict_stream(
    {"input": [{"role": "user", "content": "Pythonで7*6は何ですか？"}]}
):
    print(chunk, "-----------\n")

# COMMAND ----------

from IPython.display import Image, display

# エージェントのグラフ構造を可視化
display(Image(AGENT.agent.get_graph().draw_mermaid_png()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## エージェントをMLflowモデルとしてロギング
# MAGIC
# MAGIC `agent.py`ファイルからコードとしてエージェントをロギングします。[Databricks MCPサーバーに接続するエージェントのデプロイ](https://docs.databricks.com/aws/ja/generative-ai/mcp/managed-mcp#%E3%82%A8%E3%83%BC%E3%82%B8%E3%82%A7%E3%83%B3%E3%83%88%E3%82%92%E3%83%87%E3%83%97%E3%83%AD%E3%82%A4%E3%81%99%E3%82%8B)を参照してください。

# COMMAND ----------

import mlflow
from agent import LLM_ENDPOINT_NAME
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksFunction
from pkg_resources import get_distribution

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME), 
    DatabricksFunction(function_name="system.ai.python_exec")
]

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        resources=resources,
        pip_requirements=[
            "databricks-mcp",
            f"mlflow=={get_distribution('mlflow').version}",
            f"langgraph=={get_distribution('langgraph').version}",
            f"mcp=={get_distribution('mcp').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
        ]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Agent Evaluation](https://docs.databricks.com/aws/ja/mlflow3/genai/eval-monitor)によるエージェントの評価
# MAGIC
# MAGIC 評価データセット内のリクエストや期待される応答を編集し、エージェントを繰り返し評価できます。mlflowを活用して品質指標を追跡しましょう。
# MAGIC
# MAGIC [事前定義済みのLLMスコアラー](https://docs.databricks.com/aws/ja/mlflow3/genai/eval-monitor/predefined-judge-scorers)でエージェントを評価したり、[カスタム指標](https://docs.databricks.com/aws/ja/mlflow3/genai/eval-monitor/custom-scorers)を追加することも可能です。

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety, RetrievalRelevance, RetrievalGroundedness

eval_dataset = [
    {
        "inputs": {
            "input": [
                {
                    "role": "user",
                    "content": "15番目のフィボナッチ数を計算してください"
                }
            ]
        },
        "expected_response": "15番目のフィボナッチ数は610です。"
    }
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: AGENT.predict({"input": input}),
    scorers=[RelevanceToQuery(), Safety()], # 該当する場合は、ここに他のスコアラーを追加
)

# 評価結果をMLflow UIで確認（コンソール出力を参照）

# COMMAND ----------

# MAGIC %md
# MAGIC ## デプロイ前のエージェント検証
# MAGIC エージェントを登録・デプロイする前に、[mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) APIを使って事前検証を行いましょう。Databricksドキュメント（[AWS](https://docs.databricks.com/aws/ja/machine-learning/model-serving/model-serving-debug#%E3%83%87%E3%83%97%E3%83%AD%E3%82%A4%E5%89%8D%E3%81%AB%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E5%85%A5%E5%8A%9B%E3%82%92%E6%A4%9C%E8%A8%BC%E3%81%99%E3%82%8B) | [Azure](https://learn.microsoft.com/ja-jp/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks)）も参照してください。

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"input": [{"role": "user", "content": "Pythonで7*6は何ですか？"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalogへのモデル登録
# MAGIC
# MAGIC エージェントをデプロイする前に、MLflowモデルをUnity Catalogに登録する必要があります。
# MAGIC
# MAGIC - **TODO** 下記の`catalog`、`schema`、`model_name`を編集し、MLflowモデルをUnity Catalogに登録してください。

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: UCモデルのカタログ、スキーマ、およびモデル名を定義してください
catalog = "takaakiyayoi_catalog"
schema = "agents"
model_name = "langgraph-mcp-responses-agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# モデルをUCに登録
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## エージェントのデプロイ

# COMMAND ----------

from databricks import agents

agents.deploy(
    UC_MODEL_NAME, 
    uc_registered_model_info.version,
    # ==============================================================================
    # TODO: 下記の環境変数セクションは、カスタムMCPサーバーにOAuth/サービスプリンシパルを使用する場合のみ
    #       コメントを外して設定してください。管理MCP（デフォルト）の場合、このセクションはコメントのままにしてください。
    # ==============================================================================
    # environment_vars={
    #     "DATABRICKS_CLIENT_ID": DATABRICKS_CLIENT_ID,
    #     "DATABRICKS_CLIENT_SECRET": f"{{{{secrets/{client_secret_scope_name}/{client_secret_key_name}}}}}"
    # },
    tags = {"endpointSource": "docs"}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 次のステップ
# MAGIC
# MAGIC エージェントをデプロイした後は、AIプレイグラウンドでチャットして追加チェックを行ったり、組織内のSMEに共有してフィードバックを得たり、本番アプリケーションに組み込むことができます。Databricksドキュメント（[AWS](https://docs.databricks.com/aws/ja/generative-ai/agent-framework/deploy-agent) | [Azure](https://learn.microsoft.com/ja-jp/azure/databricks/generative-ai/agent-framework/deploy-agent)）もご参照ください。

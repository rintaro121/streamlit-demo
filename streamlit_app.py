import os
import json

import streamlit as st
import streamlit.components.v1 as components

from openai import OpenAI

from llm.v1 import first_step_llm, pre_second_step_llm, second_step_llm
import re


def extract_code_blocks(text):
    # 正規表現を使用して、```で囲まれたテキストを検索
    # pattern = r"```mermaiderDiagram(.*?)```"
    pattern = r"```mermaid\s*erDiagram\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    return matches


def mermaid(code: str) -> None:
    components.html(
        f"""
        <style>
            body {{
                background-color: black;
                color: white;  /* テキストの色を白に設定 */
            }}
            pre.mermaid {{
                background-color: black;  /* この行は必要に応じて削除または保持できます */
                color: white;
            }}
        </style>
        <pre class="mermaid">
            {code}
        </pre>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=1200,
    )


USER_NAME = "user"
ASSISTANT_NAME = "assistant"


st.set_page_config(page_title="Mermaid Generation", layout="wide")
st.title("ER Diagram Demo")
st.caption("With this demo, you can generate ER Diagram for given schema")

# チャットログを保存したセッション情報を初期化
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
# user_msg = st.chat_input("ここにメッセージを入力")

with st.sidebar:
    openai_api_key = st.text_input(
        "Open AI API Key",
        key="openapi_api_key",
        type="password",
        value=os.getenv("OPENAI_API_KEY"),
    )

uploaded_file = st.file_uploader("Choose a json file", type="json")
# user_msg = st.chat_input("ここにメッセージを入力")
if uploaded_file:
    schema = json.load(uploaded_file)

    first_step_output = first_step_llm(schema, openai_api_key)
    st.write(first_step_output)
    mermaid_table_block = extract_code_blocks(first_step_output)[0]

    pre_second_output = pre_second_step_llm(schema, openai_api_key)
    second_step_output = second_step_llm(schema, pre_second_output, openai_api_key)
    st.write(second_step_output)
    mermaid_relation_block = extract_code_blocks(second_step_output)[0]

    mermaid_for_schema = "erDiagram\n" + mermaid_table_block + "\n" + mermaid_relation_block
    st.write(mermaid_for_schema)
    mermaid(mermaid_for_schema)


# if user_msg:
#     # 以前のチャットログを表示
#     for chat in st.session_state.chat_log:
#         with st.chat_message(chat["name"]):
#             st.write(chat["msg"])
#     # 最新のメッセージを表示
#     with st.chat_message(USER_NAME):
#         st.write(user_msg)
#     # アシスタントのメッセージを表示
#     # response = response_chatgpt(user_msg)
#     response = "test text "
#     with st.chat_message(ASSISTANT_NAME):
#         assistant_msg = ""
#         assistant_response_area = st.empty()
#         assistant_response_area.write(response)
# for chunk in response:
#     if chunk.choices[0].finish_reason is not None:
#         break
#     # 回答を逐次表示
#     assistant_msg += chunk.choices[0].delta.content
#     assistant_response_area.write(assistant_msg)
# セッションにチャットログを追加
# st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
# st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_msg})

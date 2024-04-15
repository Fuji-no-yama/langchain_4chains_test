import streamlit as st
from dotenv import load_dotenv
import time
import math
import pandas as pd

import langchain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.documents.base import Document
from langchain.chains.combine_documents.stuff import (
    StuffDocumentsChain,
    create_stuff_documents_chain,
)
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables.base import RunnableSequence
from langchain.output_parsers.regex import RegexParser
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.vectorstores import FAISS

load_dotenv(override=True)  # .envファイルの中身を環境変数に設定
langchain.debug = True
filename = "hikaku_kagakuron.txt"


def initialize_vectorstore() -> (
    FAISS
):  # ベクトルDBをlangchain内のインスタンスとして作成する関数
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "/workspace/DB/vector_" + (filename.rsplit(".", 1)[0]),
        embedding,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def create_stuff_chain() -> RunnableSequence:
    prompt = ChatPromptTemplate.from_template(
        "以下のquestionにcontextで与えられた情報のみを用いて答えてください。"
        "わからない場合は「わかりません」と出力してください。"
        "context:{context}"
        "question:{question}"
    )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    stuff_chain = create_stuff_documents_chain(llm, prompt)
    return stuff_chain


def create_map_reduce_chain() -> RunnableSequence:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # 部分回答から最終回答を作成するchainの作成
    combine_prompt = ChatPromptTemplate.from_template(
        "以下のcontextはquestionへの回答の集合です。"
        "このcontextを用いて最終的なquestionへの回答を作成してください。"
        "わからない場合は「わかりません」と出力してください。"
        "context:{context}"
        "question:{question}"
    )
    combine_chain = StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=combine_prompt),
        document_variable_name="context",
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_chain,
        collapse_documents_chain=combine_chain,
        token_max=4000,
    )

    # 部分回答を生成するためのchainの作成
    map_prompt = ChatPromptTemplate.from_template(
        "以下のcontextはある文書のリストです。"
        "以下のquestionにcontextで与えられた情報のみを用いて答えてください。"
        "わからない場合は「わかりません」と出力してください。"
        "context:{context}"
        "question:{question}"
    )
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # 最終的に2つのchainを結合する
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,  # 部分回答を生成するためのchain
        reduce_documents_chain=reduce_documents_chain,  # 最終回答を作成するためのchain
        document_variable_name="context",  # mapchain中の文書を表す変数名
        return_intermediate_steps=False,  # 途中ステップを返すかどうか
    )
    return map_reduce_chain


def create_map_rerank_chain() -> RunnableSequence:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # 部分回答(ランクづけ)を生成するためのchainの作成
    map_tempelate = (
        "以下のcontextはある文書のリストです。"
        "以下のquestionにcontextで与えられた情報のみを用いて答え,"
        "その回答の自信度合いをScore:に続けて1~10の整数値で出力しなさい。"
        "context:{context}"
        "question:{question}"
    )
    output_parser = RegexParser(  # 正規表現パーサ
        regex=r"([\s\S]*?)Score:(.*)",
        output_keys=["answer", "score"],
    )
    map_prompt = PromptTemplate(
        template=map_tempelate,
        input_variables=["context", "question"],
        output_parser=output_parser,
    )
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    map_rerank_chain = MapRerankDocumentsChain(
        llm_chain=map_chain,
        document_variable_name="context",
        return_intermediate_steps=False,
        answer_key="answer",
        rank_key="score",
        verbose=True,
    )
    return map_rerank_chain


def create_refine_chain() -> RunnableSequence:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    document_prompt = PromptTemplate(  # 文書を整形するためのプロンプト(任意)
        input_variables=["page_content"], template="{page_content}"
    )
    initial_prompt = ChatPromptTemplate.from_template(  # 初回に使用されるプロンプト
        "以下のquestionにcontextで与えられた情報のみを用いて答えてください。"
        "context:{context}"
        "question:{question}"
    )
    refine_prompt = ChatPromptTemplate.from_template(  # 2回目以降に使用されるプロンプト
        "以下のquestionに対するあなたの以前の回答が「{prev_response}」です。"
        "追加の情報である以下のcontextを用いて以前の回答をより良い回答にしてください。"
        "question:{question}"
        "context:{context}"
    )
    refine_chain = RefineDocumentsChain(
        initial_llm_chain=LLMChain(llm=llm, prompt=initial_prompt),
        refine_llm_chain=LLMChain(llm=llm, prompt=refine_prompt),
        document_prompt=document_prompt,
        document_variable_name="context",
        initial_response_name="prev_response",
    )
    return refine_chain


def invoke_chain(  # 各chainごとの回答の生成と表示
    chain: RunnableSequence, type: str, context: list[Document], prompt: str
) -> tuple[str, float, float]:
    with st.chat_message("assistant"):
        with st.spinner(type + " chain..."):
            with get_openai_callback() as cb:  # 料金等集計用callback
                start_time = time.time()
                if type == "stuff":
                    tmp_result = chain.invoke({"context": context, "question": prompt})
                else:
                    tmp_result = chain.invoke(
                        {"input_documents": context, "question": prompt}
                    )["output_text"]
                end_time = time.time()
            gen_time = math.floor((end_time - start_time) * 1000) / 1000
            r1 = "**" + type + "_result**  \n"  # chainタイプの表示
            r2 = "time:" + str(gen_time) + "  \n"  # 実行時間の表示
            r3 = (
                "cost:"
                + str(cb.total_cost)
                + "$, tokens:"
                + str(cb.total_tokens)
                + ", queries:"
                + str(cb.successful_requests)
                + "  \n"
            )  # api利用情報に関する表示
            result = str(r1 + r2 + r3 + tmp_result)
        st.markdown(result)
    return result, cb.total_cost, gen_time


def culc_rate(
    stuff_c: float, map_reduce_c: float, map_rerank_c: float, refine_c: float
) -> list[float]:  # 各chainごとのコスト・時間を受け取り比率として返す関数
    map_reduce_r = math.floor((map_reduce_c / stuff_c) * 1000) / 1000
    map_rerank_r = math.floor((map_rerank_c / stuff_c) * 1000) / 1000
    refine_r = math.floor((refine_c / stuff_c) * 1000) / 1000
    return [1.0, map_reduce_r, map_rerank_r, refine_r]


def display_result(
    context: list[Document], prompt: str
) -> None:  # 結果の生成と保存までをまとめて行う関数
    stuff_result, stuff_cost, stuff_time = invoke_chain(
        st.session_state.stuff_chain, "stuff", context, prompt
    )
    map_reduce_result, map_reduce_cost, map_reduce_time = invoke_chain(
        st.session_state.map_reduce_chain, "map_reduce", context, prompt
    )
    map_rerank_result, map_rerank_cost, map_rerank_time = invoke_chain(
        st.session_state.map_rerank_chain, "map_rerank", context, prompt
    )
    refine_result, refine_cost, refine_time = invoke_chain(
        st.session_state.refine_chain, "refine", context, prompt
    )
    data = {
        "コスト($)": [stuff_cost, map_reduce_cost, map_rerank_cost, refine_cost],
        "生成時間(s)": [stuff_time, map_reduce_time, map_rerank_time, refine_time],
        "コスト(比率)": culc_rate(
            stuff_cost, map_reduce_cost, map_rerank_cost, refine_cost
        ),
        "生成時間(比率)": culc_rate(
            stuff_time, map_reduce_time, map_rerank_time, refine_time
        ),
    }
    index = ["stuff", "map_reduce", "map_rerank", "refine"]
    df = pd.DataFrame(data, index=index)
    with st.chat_message("assistant"):
        st.table(df)

    # 会話履歴用memoryへの追加
    st.session_state.chat_memory.append(
        {
            "prompt": prompt,
            "stuff": stuff_result,
            "map_reduce": map_reduce_result,
            "map_rerank": map_rerank_result,
            "refine": refine_result,
            "table": df,
        }
    )


# 以下がstreamlit用コード
st.title("chain-test-app")

st.markdown("**文書から取得するchunk数を指定してください**")
top_k = st.slider(label="top-k", value=3, min_value=1, max_value=5)
st.write(f"200文字*{top_k}chunkが文書から検索され取得されます")

if "vectorstore" not in st.session_state:  # vectorstoreを初期化
    st.session_state.vectorstore = initialize_vectorstore()

if "memory" not in st.session_state:  # memoryを初期化
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "chat_memory" not in st.session_state:  # 会話履歴表示用memory(list[dict])を初期化
    st.session_state.chat_memory = []

if "stuff_chain" not in st.session_state:  # stuffチェインの初期化
    st.session_state.stuff_chain = create_stuff_chain()

if "map_reduce_chain" not in st.session_state:  # map_reduceチェインの初期化
    st.session_state.map_reduce_chain = create_map_reduce_chain()

if "nap_rerank_chain" not in st.session_state:  # map_rerankチェインの初期化
    st.session_state.map_rerank_chain = create_map_rerank_chain()

if "refine_chain" not in st.session_state:  # refineチェインの初期化
    st.session_state.refine_chain = create_refine_chain()

for message in st.session_state.chat_memory:  # 以前の会話履歴の表示
    with st.chat_message("user"):
        st.markdown(message["prompt"])
    with st.chat_message("assistant"):
        for chain_type in list(message.keys())[1:-1]:
            st.markdown(message[chain_type])
        st.table(message["table"])


prompt = st.chat_input("質問はありますか?")  # chat入力欄の表示

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("文書検索中..."):
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )  # 指定されたtop_kを用いてretrieverを作成
        context = retriever.get_relevant_documents(prompt)  # ベクトルDBからデータ取得

    display_result(context, prompt)  # 回答の生成と表示, メモリへの保存

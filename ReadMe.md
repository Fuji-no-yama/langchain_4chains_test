# langchainを用いた4種類のchainの性能比較用デモアプリ作成
## ファイル名と概要
- add_document.py : ベクトルDB(FAISS)へ文書をベクトル化して登録するプログラム
- chain_test.py : 4種類のchainについて, 動作を実際の文書を用いて確認するデモアプリ

## 各プログラムの詳細情報

<details><summary>add_document.py</summary>

# 概要
DBディレクトリに指定した文書をベクトル化して登録するpythonプログラムです。

# 実行方法
- .envファイルを作成し, 以下の内容を記載する。
```
OPENAI_API_KEY=***
```
- プログラム中の`filepath =`の部分に追加したい文書のファイルパスを記入する
- 指定した文書が青空文庫のテキストファイルの場合はルビを削除するためのremove_ruby関数のコメントアウトを外す
- `python add_document.py`で実行する
</details>


<details><summary>chain_test.py</summary>

# 概要
以下の4つの種別のchainについてstreamlitを用いたchat形式で動作を確認できるpythonプログラムです。

- stuff
- map_reduce
- map_rerank
- refine

RAGの対象となる文書はdoc内の「hikaku_kagakuron.txt」となっています。現在は生成された回答に加えて生成時間と価格, token数などを閲覧することが可能です。

# 実行方法
- .envファイルを作成し, 以下の内容を記載する。
```
OPENAI_API_KEY=***
```
- `streamlit run chain_test.py`で実行する
- ローカルホストで起動するので[ここにアクセス](http://localhost:8501/)
- 画面上部のスライドバーで取得する文書chunk数を指定する
- RAG対象文書に記載された内容について質問を行う(「アマゾン型の研究とは何ですか?」等がおすすめ)

# 実装方法
## 実装概要
今回は主に動作するデモアプリ本体であるchain_test.pyについて実装方法を説明していきます。今回の実装に用いたライブラリは以下の通りです。

- **LLMを用いたchain作成** : [LangChain](https://github.com/langchain-ai/langchain)
- **文書保管用ベクトルDB** : [FAISS](https://github.com/facebookresearch/faiss) (LangChain内のライブラリとして利用)
- **webアプリ化** : [Streamlit](https://github.com/streamlit/streamlit)

chain_test.pyでは4種のchainを用いて著作権フリーの説明文である[「比較化学論」](https://www.aozora.gr.jp/cards/001569/files/53214_49856.html)(青空文庫より)に対する質問回答をchat形式で行えるwebアプリを起動することができます。また, 実行時間やAPI使用料金についてもモニターし比較することができます。

## 実装詳細
chain_test.pyのコード中の各関数などについて説明を行います。とりあえず実行と実験をしてみたい方は[実験と性能比較](#実験と性能比較)まで飛ばしていただいて構いません。
### ライブラリのインポート
今回使用するライブラリのインポートは以下の通りです。
```python
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
```

### 環境変数等の設定
langchainの使用にあたりopenai APIの利用が必要なため, 環境変数として設定する必要があります。まず「.env」ファイルを作成し内部に以下の形式でopenai APIキーを記載します。
```
OPENAI_API_KEY=***
```
次に, 記載した内容を環境変数に埋め込み各種の設定を行います。`langchain.debug`の値は`True`にするとターミナルにLLMとのやり取りが全て表示されるため, 確認が不要であれば`False`にしてください。
```python
load_dotenv(override=True)  # .envファイルの中身を環境変数に設定
langchain.debug = True
filename = "hikaku_kagakuron.txt"
```

### ベクトルDBのセットアップ関数
リポジトリ内部にエンべディングされたDBは既に存在しているため, プログラム内部で使用できるインスタンスの形式でセットアップを行う関数です。
```python
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
```
### stuff chainの作成関数
stuff chainを作成する関数です。外部から与えた文書群が`{context}`に, ユーザが与えた質問が`{question}`に埋め込まれてLLMに渡される形式です。
```python
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
```

### map_reduce chainの作成関数
map reduce形式のchainを作成する関数です。主に各文書に適応されるmap_chainとそれらの出力をまとめて最終回答を作成するcombine_chainの2種類によって構成されるため, 2種類のプロンプトを用意する必要があります。
```python
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
```

### map_rerank chainの作成関数
map rerank形式のchainを作成する関数は以下の通りです。今回も各文書に対して部分回答を作成するためのmap_chainを作成していますが, 最終回答の作成は`MapRerankDocumentsChain`の内部でLLMが出力した順位を元に自動で行われます。一方で, その順位を出力させるためにLLMへのプロンプトはやや特殊になっており, その出力を正規表現で回答本体と順位へと分割するパーサを用意する必要があります。
```python
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
```

### refine chainの作成関数
refine形式のchainを作成する関数は以下の通りとなっています。refineチェインでは文書を再起的に回答と一緒にLLMに渡す都合上, 初回のみプロンプトが特殊になります。(前回回答がないため) よって, この関数内でもプロンプトは2種類用意しています。
```python
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
```
### chainの起動から結果表示までの関数
以下のコード内の各関数の役割は以下の通りです。

- **invoke_chain** : 4種類のchainをそれぞれ起動するための関数
- **culc_rate** : 実行時間と使用API料金の比をstuffチェインを基準に計算する関数
- **diplay_result** : chainの実行から生成結果, 生成時間, 使用API料金などをstreamlitの画面に出力する関数

```python
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
```
### streamlitを用いての全体の処理(main)
以上で定義した関数群を用いて全体の処理を記述するコードは以下の通りとなっています。基本的にはstreamlitのセッション機能を用いて上の関数で作成したオブジェクトを保持しながら, chat欄にユーザからの入力があった場合には各チェインを起動させて結果を表示する仕組みとなっています。  
また, ベクトルDBから何個分の文章を持ってくるのかについても指定することができるようになっています。
```python
# 以下がstreamlit用コード
st.title("chain-test-app")

st.markdown("**文書から取得するchunk数を指定してください**")
top_k = st.slider(label="top-k", value=5, min_value=1, max_value=10)
st.write(f"400文字*{top_k}chunkが文書から検索され取得されます")

if "vectorstore" not in st.session_state:  # vectorstoreを初期化
    st.session_state.vectorstore = initialize_vectorstore()

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
```

</details>
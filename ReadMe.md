# langchainを用いた4種類のchainの性能比較用デモアプリ作成
## ファイル名と概要
- add_document.py : ベクトルDB(FAISS)へ文書をベクトル化して登録するプログラム
- chain_test.py : 4種類のchainについて, 動作を実際の文書を用いて確認するデモアプリ

## 各プログラムの概要と実行方法

<details><summary>add_document.py</summary>

### 概要
DBディレクトリに指定した文書をベクトル化して登録するpythonプログラムです。

### 実行方法
- .envファイルを作成し, 以下の内容を記載する。
```
OPENAI_API_KEY=***
```
- プログラム中の`filepath =`の部分に追加したい文書のファイルパスを記入する
- 指定した文書が青空文庫のテキストファイルの場合はルビを削除するためのremove_ruby関数のコメントアウトを外す
- `python add_document.py`で実行する
</details>


<details><summary>chain_test.py</summary>

### 概要
以下の4つの種別のchainについてstreamlitを用いたchat形式で動作を確認できるpythonプログラムです。

- stuff
- map_reduce
- map_rerank
- refine

RAGの対象となる文書はdoc内の「hikaku_kagakuron.txt」となっています。現在は生成された回答に加えて生成時間と価格, token数などを閲覧することが可能です。

### 実行方法
- .envファイルを作成し, 以下の内容を記載する。
```
OPENAI_API_KEY=***
```
- `streamlit run chain_test.py`で実行する
- ローカルホストで起動するので[ここにアクセス](http://localhost:8501/)
- 画面上部のスライドバーで取得する文書chunk数を指定する
- RAG対象文書に記載された内容について質問を行う(「アマゾン型の研究とは何ですか?」等がおすすめ)

</details>
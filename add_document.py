# docフォルダ内にある文書をベクトルデータベース(FAISS)としてDBディレクトリに保存する関数
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import re
import os

load_dotenv(override=True)  # .envファイルの中身を環境変数に設定
filename = "hikaku_kagakuron.txt"


def remove_ruby(
    filename: str
) -> None:  # 指定したテキストファイルの余計な部分を正規表現で除去しエンコードをshiftjis->utf8へ変更する関数(青空文庫ファイル専用)
    filepath = "/workspace/doc/" + filename
    with open(filepath, "r", encoding="utf_8") as input_file:
        content = input_file.read()
    pattern = r"《[^》]*》"
    tmp = re.sub(pattern, "", content)
    pattern = r"-{2,}.*?-{2,}"
    tmp = re.sub(pattern, "", tmp, flags=re.DOTALL)
    pattern = r"［＃.*\n"
    modified_content = re.sub(pattern, "", tmp)
    with open(filepath, "w", encoding="utf-8") as output_file:
        output_file.write(modified_content)


def setup_vectorDB(
    filename: str
) -> None:  # ファイル名を指定してその青空文庫ファイルをローカルのvectorDBに格納する関数(フォルダがない場合に一度だけ呼び出される)
    loader = TextLoader("/workspace/doc/" + filename)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    dummy_text, dummy_id = "1", 1
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts([dummy_text], embeddings, ids=[dummy_id])
    vectorstore.delete([dummy_id])
    vectorstore.merge_from(FAISS.from_documents(docs, embeddings))
    dir_name = "/workspace/DB/vector_" + (filename.rsplit(".", 1)[0])
    vectorstore.save_local(dir_name)
    print("作成が完了しました")


if __name__ == "__main__":
    remove_ruby(filename)
    if not os.path.exists("/workspace/DB/vector_" + (filename.rsplit(".", 1)[0])):
        setup_vectorDB(filename)
    else:
        print("指定されたファイルはすでにDB内に存在しています")

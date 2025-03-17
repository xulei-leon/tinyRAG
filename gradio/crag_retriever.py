import os
import tomllib
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.base import VectorStore
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredHTMLLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class CragRetriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        persist_directory: str = "vector_store",
    ):
        os.makedirs(persist_directory, exist_ok=True)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"  # to avoid downloading models
        os.environ["HF_DATASETS_OFFLINE"] = "1"  # to avoid downloading datasets

        self.batch_size = 10
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "batch_size": self.batch_size,
                "convert_to_numpy": True,
                "device": "cpu",
            },
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=[
                "\n\n",
                "\n",
                ".",
                "!",
                "?",
                ";",
                ":",
                "\t",
                "\r\n",
                "\r",
                "，",
                "。",
                "！",
                "？",
                "；",
                "：",
            ],
        )

        self.vector_store: VectorStore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            collection_name="calerie-health",
            client_settings=ChromaSettings(
                anonymized_telemetry=False,  # reduce the memory usage
                allow_reset=False,  # prevent accidental deletion of the vector store
            ),
        )

        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,  # number of results to return
                "lambda_mult": 0.7,  # lambda value for MMR
                "score_threshold": 0.3,  # similarity threshold
                # "filter": {"my_key": "my_value"}, # filter the results based on a key-value pair
            },
        )

    # create vector store
    def create_vector(self, files_directory: str):
        """Add all files from a given path to the vector store using DirectoryLoader."""

        try:
            loader = DirectoryLoader(
                files_directory,
                glob="**/*",
                loader_cls=UnstructuredFileLoader,
                loader_kwargs={
                    "mode": "single",
                    "strategy": "fast",
                    "autodetect_encoding": True,
                },
                use_multithreading=True,
                # max_concurrency=2,
                show_progress=True,
                recursive=True,
                silent_errors=True,
            )
            documents = loader.load()
            print(f"Loaded {len(documents)} documents.")
        except Exception as e:
            print(f"Load {files_directory} error.")

        all_splits = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(documents=all_splits)
        print(f"Added {len(all_splits)} document splits to the vector store.")

    # query vector store
    def query(self, query: str, top_k: int = 1):
        """Query the vector store with a text and return the top_k results."""
        # return self.vector_store.query(query, top_k=top_k)
        return self.retriever.invoke(query, top_k=top_k)


import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="CRAG command",
        epilog="Example:\n"
        "  create: app.py --create ./files\n"
        "  query: app.py --query 'My question?'",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-c", "--create", metavar="PATH", help="Create veator from files"
    )
    group.add_argument("-q", "--query", metavar="QUESTION", help="Query")

    args = parser.parse_args()
    if args.create:
        if not os.path.isdir(args.create):
            parser.error(f"Do not exist: {args.create}")

    return args


def main():
    try:
        args = parse_args()
    except KeyboardInterrupt:
        print("\nError: KeyboardInterrupt")
        sys.exit(130)
    except Exception as e:
        sys.exit(2)

    print("=== Init config ===")
    with open("config.toml", "rb") as f:
        config_data = tomllib.load(f)
        model_name = config_data.get("huggingface", {}).get("embed_model")

    root_path = "./var"
    files_directory = os.path.join(root_path, "files")
    persist_directory = os.path.join(root_path, "vector_store")

    print(f"model_name: {model_name}")
    print(f"files_directory: {files_directory}")
    print(f"persist_directory: {persist_directory}")

    print("=== Init retriever ===")
    retriever = CragRetriever(
        model_name=model_name,
        persist_directory=persist_directory,
    )

    if args.create:
        print("=== Create retriever ===")
        retriever.create_vector(files_directory=args.create)
    elif args.query:
        print("=== Query retriever ===")
        query = args.query.encode("utf-8").decode("utf-8")
        print(f"Question: {query}\n")

        results = retriever.query(query, top_k=3)
        if not results:
            print("=== No results ===")
            return

        for doc in results:
            print(" = Answer =")
            print(doc.page_content)
            print("\n")
    else:
        print("=== No action ===")


if __name__ == "__main__":
    main()

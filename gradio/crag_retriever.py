###############################################################################
# Imports
###############################################################################
import os
import sys
import argparse
import tomllib
from typing import List

# langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.base import VectorStore
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredFileLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class CragRetriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        persist_directory: str = "vector_store",
    ):
        os.makedirs(persist_directory, exist_ok=True)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"  # to avoid downloading models
        # to avoid downloading datasets
        os.environ["HF_DATASETS_OFFLINE"] = "1"

        self.batch_size = 10
        self.embeddings = self.__init_embedder(model_name)
        self.text_splitter = self.__init_text_splitter()
        self.vector_store = self.__init_vector_store(persist_directory)
        self.retriever = self.__init_retriever()

    # create vector store
    def build_index(self, files_directory: str):
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
    def invoke(self, query: str) -> List[Document]:
        """Query the vector store with a text and return the top_k results."""
        return self.retriever.invoke(query)

    ###############################################################################
    # Internal functions
    ###############################################################################
    def __init_embedder(self, model_name: str):
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "batch_size": self.batch_size,
                "convert_to_numpy": True,
                "device": "cpu",
            },
        )

    def __init_text_splitter(self):
        return RecursiveCharacterTextSplitter(
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

    def __init_vector_store(self, persist_directory: str):
        return Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            collection_name="calerie-health",
            client_settings=ChromaSettings(
                anonymized_telemetry=False,  # reduce the memory usage
                allow_reset=False,  # prevent accidental deletion of the vector store
            ),
        )

    def __init_retriever(self):
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,  # number of results to return
                "lambda_mult": 0.7,  # lambda value for MMR
                "score_threshold": 0.5,  # similarity threshold
            },
        )

################################################################################
# main
################################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        prog="crag_retriever.py",
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
        files_directory = config_data.get("vector", {}).get("files_directory")
        persist_directory = config_data.get("vector", {}).get("persist_directory")

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
        retriever.build_index(files_directory=args.create)
    elif args.query:
        print("=== Query retriever ===")
        query = args.query.encode("utf-8").decode("utf-8")
        print(f"Question: {query}\n")

        results = retriever.invoke(query)
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

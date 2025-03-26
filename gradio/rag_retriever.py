###############################################################################
# Imports
###############################################################################
import os
import sys
import argparse
import tomllib
from typing import List
import pickle

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
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# TODO List
#
class RagRetriever:
    def __init__(
        self,
        embed_model: str = "BAAI/bge-small-zh-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        persist_directory: str = "persist",
        collection_name: str = "rag",
    ):
        os.makedirs(persist_directory, exist_ok=True)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"  # to avoid downloading models
        os.environ["HF_DATASETS_OFFLINE"] = "1"  # to avoid downloading datasets

        self.embed_model = embed_model
        self.reranker_model = reranker_model

        # Vector store directory
        self.vector_store_directory = os.path.join(persist_directory, "vector")
        os.makedirs(self.vector_store_directory, exist_ok=True)

        # BM25 index file
        bm25_directory = os.path.join(persist_directory, "bm25")
        os.makedirs(bm25_directory, exist_ok=True)
        self.bm25_index_file = os.path.join(bm25_directory, "bm25_index.pkl")

        self.collection_name = collection_name
        self.search_type = "mmr"

        self.batch_size = 10
        self.chunk_size = 1000
        self.chunk_overlap = 150
        self.k = 3
        self.score_threshold = 0.55

        self.embeddings = self.__init_embedder()
        self.text_splitter = self.__init_text_splitter()
        self.vector_store = self.__init_vector_store()
        self.vector_retriever = None
        self.bm25_retriever = None
        self.retriever = None
        self.rerank_retriever = None

    # create vector store
    def build_index(self, files_directory: str):
        documents = self.__files_load(files_directory)
        if not documents:
            print("No documents found. Skipping vector store creation.")
            return

        self.__build_vector(documents)
        self.__build_bm25(documents)

    def load_index(self):
        self.vector_retriever = self.__init_vector_retriever()
        self.bm25_retriever = self.__init_bm25_retriever()

    # query vector store
    def query(self, query: str) -> List[Document]:
        if not self.retriever:
            self.retriever = self.__init_retriever()

        return self.retriever.invoke(query)

    def query_rerank(self, query: str) -> List[Document]:
        if not self.retriever:
            self.retriever = self.__init_retriever()

        if not self.rerank_retriever:
            self.rerank_retriever = self.__init_rerank_retriever()

        return self.rerank_retriever.invoke(query)

    ###############################################################################
    # Internal functions
    ###############################################################################
    def __files_load(self, files_directory: str) -> List[Document]:
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
            documents = None

        return documents

    def __build_vector(self, documents: List[Document]):
        all_splits = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(documents=all_splits)
        print(f"Added {len(all_splits)} document splits to the vector store.")

    def __build_bm25(self, documents: List[Document]):
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = self.k * 2
        with open(self.bm25_index_file, "wb") as f:
            pickle.dump(self.bm25_retriever, f)
        print(f"Added {len(documents)} documents to the bm25 index.")

    def __init_embedder(self):
        return HuggingFaceEmbeddings(
            model_name=self.embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "batch_size": self.batch_size,
                "convert_to_numpy": True,
                "device": "cpu",
            },
        )

    def __init_text_splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
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

    def __init_vector_store(self):
        return Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.vector_store_directory,
            collection_name=self.collection_name,
            client_settings=ChromaSettings(
                anonymized_telemetry=False,  # reduce the memory usage
                allow_reset=False,  # prevent accidental deletion of the vector store
            ),
        )

    def __init_vector_retriever(self):
        if self.search_type == "mmr":
            search_type = "mmr"
            search_kwargs = {
                "k": self.k * 2,  # number of results to return
                "score_threshold": self.score_threshold,
                "fetch_k": self.k * 10,
                "lambda_mult": 0.9,  # lambda value for MMR
            }
        else:
            search_type = "similarity_score_threshold"
            search_kwargs = {
                "k": self.k * 2,
                "score_threshold": self.score_threshold,
            }

        try:
            return self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs,
            )
        except Exception as e:
            print(f"Load vector_store error: {e}")
            return None

    def __init_bm25_retriever(self):
        try:
            with open(self.bm25_index_file, "rb") as f:
                bm25_retriever = pickle.load(f)
                print(f"Load bm25_index_file: {self.bm25_index_file}")
        except FileNotFoundError:
            print(f"Do not found bm25_index_file: {self.bm25_index_file}")
            bm25_retriever = None
        except Exception as e:
            print(f"Load bm25_index_file error: {e}")
            bm25_retriever = None

        return bm25_retriever

    def __init_retriever(self):
        # Initialize ensemble retriever after both retrievers are inited
        return EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.7, 0.3],
            lmbda=0.6,
            rerank_k=self.k,
            return_sources=True,
            deduplicate=True,
        )

    def __init_rerank_retriever(self):
        model = HuggingFaceCrossEncoder(
            model_name=self.reranker_model, model_kwargs={"device": "cpu"}
        )
        compressor = CrossEncoderReranker(model=model, top_n=self.k)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever,
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
        "-c", "--create", metavar="PATH", help="Create vector from files"
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
    retriever = RagRetriever(
        embed_model=model_name,
        persist_directory=persist_directory,
    )

    if args.create:
        print("=== Create retriever ===")
        retriever.build_index(files_directory=args.create)
    elif args.query:
        retriever.load_index()
        print("=== Query retriever ===")
        query = args.query.encode("utf-8").decode("utf-8")
        print(f"Question: {query}\n")

        results = retriever.query(query)
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

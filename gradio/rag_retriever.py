###############################################################################
# Imports
###############################################################################
import os
import sys
import argparse
import tomllib
from typing import List
import pickle
from dotenv import load_dotenv
import nltk
from chromadb.config import Settings as ChromaSettings

# langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.vectorstores.base import VectorStore
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# my modules
from rag_file_loader import RagFileLoader


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
        os.environ["TRANSFORMERS_OFFLINE"] = "1"  # to avoid downloading models
        os.environ["HF_DATASETS_OFFLINE"] = "1"  # to avoid downloading datasets
        with open("config.toml", "rb") as f:
            config_data = tomllib.load(f)
            self.split_chunk_size = config_data.get("retriever", {}).get(
                "split_chunk_size", 1000
            )
            self.split_chunk_overlap = config_data.get("retriever", {}).get(
                "split_chunk_overlap", 150
            )
            self.search_result_num = config_data.get("retriever", {}).get(
                "search_result_num", 3
            )
            self.score_threshold = config_data.get("retriever", {}).get(
                "score_threshold", 0.55
            )

        os.makedirs(persist_directory, exist_ok=True)

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

        # init methods
        self.file_loader = RagFileLoader()

        self.embeddings = self.__init_embedder()
        self.text_splitter = self.__init_text_splitter()
        self.vector_store = self.__init_vector_store()
        self.vector_retriever = None
        self.bm25_retriever = None
        self.retriever = None
        self.reranker = self.__init_reranker()
        self.rerank_retriever = None

    # create vector store
    def build_index(self, directory: str):
        documents = self.__files_load(directory)
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

    def query_vector(self, query: str) -> List[Document]:
        return self.vector_retriever.invoke(query)

    def query_bm25(self, query: str) -> List[Document]:
        return self.bm25_retriever.invoke(query)

    def query_rerank(self, query: str) -> List[Document]:
        if not self.retriever:
            self.retriever = self.__init_retriever()

        if not self.rerank_retriever:
            self.rerank_retriever = self.__init_rerank_retriever()

        return self.rerank_retriever.invoke(query)

    def query_score(self, question: str, context: str) -> float:
        pairs = [(question, context)]
        scores = self.reranker.score(pairs)
        return round(scores[0], 2)

    ###############################################################################
    # Internal functions
    ###############################################################################
    def __files_load(self, directory: str) -> List[Document]:
        try:
            documents = RagFileLoader.load(directory=directory)
            if documents:
                print(f"Loaded {directory} {len(documents)} documents.")
        except Exception as e:
            print(f"Load {directory} except {e}.")
            documents = []

        return documents

    def __build_vector(self, documents: List[Document]):
        all_splits = self.text_splitter.split_documents(documents)
        filtered_splits = filter_complex_metadata(all_splits)
        self.vector_store.add_documents(documents=filtered_splits)
        print(f"Added {len(all_splits)} document splits to the vector store.")

    def __build_bm25(self, documents: List[Document]):
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = self.search_result_num * 2
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
            chunk_size=self.split_chunk_size,
            chunk_overlap=self.split_chunk_overlap,
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
                " ",
                "",
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
                "k": self.search_result_num * 2,  # number of results to return
                "score_threshold": self.score_threshold,
                "fetch_k": self.search_result_num * 10,
                "lambda_mult": 0.9,  # lambda value for MMR
            }
        else:
            search_type = "similarity_score_threshold"
            search_kwargs = {
                "k": self.search_result_num * 2,
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
            rerank_k=self.search_result_num,
            return_sources=True,
            deduplicate=True,
        )

    def __init_reranker(self):
        return HuggingFaceCrossEncoder(
            model_name=self.reranker_model, model_kwargs={"device": "cpu"}
        )

    def __init_rerank_retriever(self):
        compressor = CrossEncoderReranker(
            model=self.reranker, top_n=self.search_result_num
        )
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
    # if args.create:
    #     if not os.path.isdir(args.create):
    #         parser.error(f"Do not exist: {args.create}")

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
        embed_model = config_data.get("huggingface", {}).get("embed_model")
        reranker_model = config_data.get("huggingface", {}).get("reranker_model")
        files_directory = config_data.get("vector", {}).get("files_directory")
        persist_directory = config_data.get("vector", {}).get("persist_directory")
        collection_name = config_data.get("vector", {}).get("collection_name")

    print(f"model_name: {embed_model}")
    print(f"files_directory: {files_directory}")
    print(f"persist_directory: {persist_directory}")

    print("=== Init retriever ===")
    retriever = RagRetriever(
        embed_model=embed_model,
        reranker_model=reranker_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    if args.create:
        print("=== Create retriever ===")
        directory = args.create or files_directory
        retriever.build_index(directory=directory)
    elif args.query:
        retriever.load_index()
        print("=== Query retriever ===")
        query = args.query.encode("utf-8").decode("utf-8")
        print(f"Question: {query}\n")

        results = retriever.query_vector(query)
        if results:
            for doc in results:
                print(" = query_vector results =")
                print(doc.page_content)
                print("\n")

        results = retriever.query_bm25(query)
        if results:
            for doc in results:
                print(" = query_bm25 results =")
                print(doc.page_content)
                print("\n")

        results = retriever.query(query)
        if results:
            for doc in results:
                print(" = query results =")
                print(doc.page_content)
                print("\n")

    else:
        print("=== No action ===")


if __name__ == "__main__":
    main()

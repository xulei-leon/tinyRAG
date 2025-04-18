# Build Knowledge Base

import os
import sys
import argparse
import tomllib

# my modules
from src.rag_retriever import RagRetriever


def build_index(args, files_directory):
    print("=== Create bm25 ===")
    directory = args.build_index or files_directory
    retriever.build_bm25(directory=directory)
    print("=== Create end ===")


def build_vector(args, files_directory):
    print("=== Create vector ===")
    directory = args.build_vector or files_directory
    retriever.build_vector(directory=directory)
    print("=== Create end ===")


def build_graph(args, files_directory):
    print("=== Create graph ===")
    print("=== TODO ===")


def handle_show(args, files_directory):
    print("=== Show documents ===")
    directory = args.show or files_directory
    retriever.show_document(directory=directory)


def query_index(args):
    retriever.load_index(_with_directory)
    print("=== Query retriever ===")
    query = args.query.encode("utf-8").decode("utf-8")
    print(f"Question: {query}\n")

    results = retriever.query_bm25(query)
    if results:
        for doc in results:
            print(" = query_bm25 results =")
            print(doc.page_content)
            print("\n")


def query_vector(args):
    retriever.load_index(_with_directory)
    print("=== Query retriever ===")
    query = args.query.encode("utf-8").decode("utf-8")
    print(f"Question: {query}\n")

    results = retriever.query_vector(query)
    if results:
        for doc in results:
            print(" = query_vector results =")
            print(doc.page_content)
            print("\n")


def query_graph(args):
    print("=== Query graph ===")
    print("=== TODO ===")


################################################################################
# Init config
################################################################################
print("=== Init config ===")
with open("config/config.toml", "rb") as f:
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


################################################################################
# main
################################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        prog="build_kb.py",
        description="Build Knowledge Base Tool",
        epilog="Example:\n"
        "  build bm25 index: app.py --build_index ./files\n"
        "  query bm25 index: app.py --query_index 'My question?'\n"
        "  build vector : app.py --build_vector ./files\n"
        "  query vector: app.py --query_vector 'My question?'\n"
        "  build graph: app.py --build_graph ./files\n"
        "  query graph: app.py --query_graph 'My question?'\n"
        "  show: app.py --show ./files\n",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i", "--build_index", metavar="PATH", help="Build BM25 index from directory"
    )
    group.add_argument(
        "-v", "--build_vector", metavar="PATH", help="Build vector from directory"
    )
    group.add_argument(
        "-g", "--build_graph", metavar="PATH", help="Build graph from directory"
    )

    group.add_argument(
        "-n", "--query_index", metavar="QUESTION", help="Query from BM25 index"
    )
    group.add_argument(
        "-e", "--query_vector", metavar="QUESTION", help="Query from vector"
    )
    group.add_argument(
        "-r", "--query_graph", metavar="QUESTION", help="Query from graph"
    )

    group.add_argument(
        "-s", "--show", metavar="PATH", help="Show document content from directory"
    )

    args = parser.parse_args()
    return args


def main():
    try:
        args = parse_args()
    except KeyboardInterrupt:
        print("\nError: KeyboardInterrupt")
        sys.exit(130)
    except Exception as e:
        sys.exit(2)

    if args.build_index:
        build_index(args, files_directory)
    elif args.build_vector:
        build_vector(args, files_directory)
    elif args.build_graph:
        build_graph(args, files_directory)
    elif args.show:
        handle_show(args, files_directory)
    elif args.query:
        handle_query(args)
    else:
        print("=== No action ===")


if __name__ == "__main__":
    main()

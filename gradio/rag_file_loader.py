###############################################################################
# Imports
###############################################################################
import os
from typing import List
from unstructured.documents.elements import Element
from unstructured.cleaners.core import (
    replace_unicode_quotes,
    clean_non_ascii_chars,
    clean_extra_whitespace,
)

# langchain
from langchain.schema import Document
from langchain_unstructured.document_loaders import UnstructuredLoader


#
class RagFileLoader:
    @classmethod
    def load(cls, directory: str) -> list[Document]:
        return cls.load_as_single(directory=directory)

    @classmethod
    def load_as_single(cls, directory: str) -> list[Document]:
        return cls.__load_directory(directory=directory, mode="basic")

    @classmethod
    def load_as_chunk(cls, directory: str) -> list[Document]:
        return cls.__load_directory(directory=directory, mode=None)

    @classmethod
    def __load_directory(cls, directory: str, mode: str) -> list[Document]:
        documents = []
        for root, _, files in os.walk(directory):
            file_paths = []
            for file in files:
                path = os.path.join(root, file)
                file_paths.append(path)

            if not file_paths:
                continue

            docs = cls.__load_files(file_paths=file_paths, mode=mode)
            if docs:
                print(f"load {root} files {len(docs)}.")
                documents.extend(docs)

        return documents

    @classmethod
    def __load_files(cls, file_paths: [], mode: str) -> list[Document]:
        try:
            loader = UnstructuredLoader(
                file_path=file_paths,
                chunking_strategy=mode,
                max_characters=1000000,
                include_orig_elements=False,
                post_processors=[
                    replace_unicode_quotes,
                    clean_non_ascii_chars,
                    clean_extra_whitespace,
                    cls.__custom_cleaner,
                ],
            )
            documents = loader.load()
        except Exception as e:
            print(f"load {root} except: {e}.")
            documents = []

        return documents

    @staticmethod
    def __custom_cleaner(elements: List) -> List:
        # Now do nothing
        return elements

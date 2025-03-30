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
    @staticmethod
    def list_files(directory: str) -> tuple[list[str], int]:
        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                path = os.path.join(root, file)
                all_files.append(path)
        return all_files

    @classmethod
    def load_files(cls, file_paths: []):
        return cls.__load_files(file_paths=file_paths, mode="basic")

    @classmethod
    def load_directory(cls, directory: str) -> list[Document]:
        return cls.__load_directory(directory=directory, mode="basic")

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
        documents = []
        for file_path in file_paths:
            if not cls.__is_text_file(file_path):
                continue

            try:
                loader = UnstructuredLoader(
                    file_path=file_path,
                    chunking_strategy=mode,
                    max_characters=1000000,
                    include_orig_elements=False,
                    post_processors=[
                        replace_unicode_quotes,
                        # clean_non_ascii_chars, ## Warning: this will remove all chinese characters
                        clean_extra_whitespace,
                        cls.__custom_cleaner,
                    ],
                )
                document = loader.load()
                documents.extend(document)
            except Exception as e:
                print(f"load {file_path} except: {e}.")
                continue

        return documents

    @staticmethod
    def __custom_cleaner(elements: List) -> List:
        # Now do nothing
        return elements

    @staticmethod
    def __is_text_file(filename):
        allowed_extensions = [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".txt", ".html"]
        _, ext = os.path.splitext(filename)
        return ext.lower() in allowed_extensions

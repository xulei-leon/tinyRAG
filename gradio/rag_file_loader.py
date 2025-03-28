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
#
class RagFileLoader:
    @classmethod
    def load(cls, directory: str) -> list[Document]:
        documents = []
        for root, _, files in os.walk(directory):
            # print(f"load root:{root}\nfiles:{files}.")
            if not files:
                continue

            file_paths = []
            for file in files:
                path = os.path.join(root, file)
                file_paths.append(path)
            # print(f"file_paths: {file_paths}.")

            if not file_paths:
                continue

            try:
                loader = UnstructuredLoader(
                    file_path=file_paths,
                    #                    post_processors=[
                    #                        "replace_unicode_quotes",
                    #                        "clean_non_ascii_chars",
                    #                        "clean_extra_whitespace",
                    #                        cls.__custom_cleaner,
                    #                    ],
                )
                docs = loader.load()
                print(f"load {root} files {len(docs)}.")
                documents.extend(docs)
            except Exception as e:
                print(f"load {root} except: {e}.")
                continue

        return documents

    @staticmethod
    def __custom_cleaner(elements: List[Element]) -> List[Element]:
        processed_elements = []

        for element in elements:
            if hasattr(element, "text"):
                element.text = element.text.lower()

            processed_elements.append(element)

        return processed_elements

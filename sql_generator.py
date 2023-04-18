import os

import sentence_transformers
from langchain import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings

import logging
logging.basicConfig(level=logging.INFO)
EMBEDDING_DEVICE = "cpu"
embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese", )
embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name,
                                                              device=EMBEDDING_DEVICE)


def init_knowledge_vector_store(filepath: str):
    if not os.path.exists(filepath):
        print("路径不存在")
        return None
    elif os.path.isfile(filepath):
        file = os.path.split(filepath)[-1]
        try:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            docs = loader.load()
            print(f"{file} 已成功加载")
        except:
            print(f"{file} 未能成功加载")
            return None
    elif os.path.isdir(filepath):
        docs = []
        for file in os.listdir(filepath):
            fullfilepath = os.path.join(filepath, file)
            try:
                loader = UnstructuredFileLoader(fullfilepath, mode="elements")
                docs += loader.load()
                print(f"{file} 已成功加载")
            except:
                print(f"{file} 未能成功加载")

    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


vector_store = init_knowledge_vector_store("/content/knowledge1.txt")
logging.warning("similarity_search:{}".format(vector_store.similarity_search("中国流行的美国发钱的段子")))
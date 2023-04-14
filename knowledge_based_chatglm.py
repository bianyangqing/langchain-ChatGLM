from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from chatglm_llm import ChatGLM
import sentence_transformers
import torch
import os
import gradio as gr
import readline
import logging
logging.basicConfig(level=logging.INFO)

NOTE = 'V1版本局限性说明 \n' \
       '1、大模型接口调用较慢，目前流式打字机效果还在调试中，请耐心等待。（如果我们明天打字机出不来的话）\n' \
       '2、目前饿了么商家知识库数据量还非常小，可能会出现“幻觉”现象并返回不符合事实的信息（尤其是页面配置、url等）。\n' \
       '3、V2版本计划提升回答的准确性、其他用户体验等。'

# Global Parameters
EMBEDDING_MODEL = "text2vec"
VECTOR_SEARCH_TOP_K = 6
LLM_MODEL = "chatglm-6b-int4"
LLM_HISTORY_LEN = 3
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Show reply with source text from input document
REPLY_WITH_SOURCE = True

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

llm_model_dict = {
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b": "THUDM/chatglm-6b",
}

VECTOR_SEARCH_TOP_K = 6
chatglm = ChatGLM()
chatglm.load_model(model_name_or_path=llm_model_dict[LLM_MODEL])
chatglm.history_len = LLM_HISTORY_LEN
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL], )
embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name,
                                                              device=DEVICE)


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


vector_store_result = init_knowledge_vector_store(os.environ.get('file_path'))


def get_knowledge_based_answer(query, chat_history=[]):
    global chatglm, embeddings, vector_store_result

    logging.warning("query:{}".format(query))
    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

已知内容:
{context}

问题:
{question}"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chatglm.history = chat_history
    knowledge_chain = RetrievalQA.from_llm(
        llm=chatglm,
        retriever=vector_store_result.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K}),
        prompt=prompt
    )
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )

    knowledge_chain.return_source_documents = True

    result = knowledge_chain({"query": query})
    logging.warning("result:{}".format(result))
    chatglm.history[-1][0] = query
    return result, chatglm.history


with gr.Blocks(css="#chatbot{height:600px} .overflow-y-auto{height:500px}",
               title="商家小助手",
               description=NOTE,
               ) as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", label="Question").style(
            container=False)

    txt.submit(get_knowledge_based_answer, [txt, state], [chatbot, state])

demo.launch(share=True, inbrowser=True)

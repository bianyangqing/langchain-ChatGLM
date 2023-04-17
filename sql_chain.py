from langchain import OpenAI, SQLDatabase, SQLDatabaseChain, PromptTemplate
import os
from chatglm_llm import ChatGLM
import gradio as gr

MODEL_NAME_GPT = "chatGpt-3.5"

MODEL_NAME_GLM = "ChatGLM-6B-int4"

_DEFAULT_TEMPLATE = """Given an input question,

create a syntactically correct {dialect} query to run

Only return the query without other words

Only use the tables listed below.

{table_info}

Question: {input}"""

PROMPT_SIMPLE = PromptTemplate(
    input_variables=["input", "table_info", "dialect"],
    template=_DEFAULT_TEMPLATE,
)

db = SQLDatabase.from_uri("sqlite:///test.db")
llm = OpenAI(temperature=0)


chatglm = ChatGLM()
chatglm.load_model(model_name_or_path="THUDM/chatglm-6b-int4")
chatglm.history_len = 3


db_chain_glm = SQLDatabaseChain(llm=chatglm, database=db,
                                prompt = PROMPT_SIMPLE, verbose=True)


db_chain_gpt = SQLDatabaseChain(llm=llm, database=db, verbose=True)



def get_knowledge_based_answer(query, model_name, chat_history=[]):
    if model_name == MODEL_NAME_GLM:
        result = db_chain_glm.run(query)
    else:
        result = db_chain_gpt.run(query)
    print("final_result:{}".format(result))
    return [(query,result)], [query,result]



with gr.Blocks(css="#chatbot{height:600px} .overflow-y-auto{height:500px}"
               ) as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])
    with gr.Row():
        model_name = gr.inputs.Radio([MODEL_NAME_GPT, MODEL_NAME_GLM], default=MODEL_NAME_GLM, label="Model")

        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", label="Question").style(
            container=False)


    txt.submit(get_knowledge_based_answer, [txt, model_name, state], [chatbot, state])

demo.launch(share=True, inbrowser=True)

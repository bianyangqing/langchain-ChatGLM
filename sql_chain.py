from langchain import OpenAI, SQLDatabase, SQLDatabaseChain, PromptTemplate
import os
from chatglm_llm import ChatGLM
import gradio as gr



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


db_chain = SQLDatabaseChain(llm=chatglm, database=db,
                            prompt = PROMPT_SIMPLE, verbose=True)



def get_knowledge_based_answer(query, chat_history=[]):
    result = db_chain.run(query)
    print("final_result:{}".format(result))
    return [(query,result)], [query,result]



with gr.Blocks(css="#chatbot{height:600px} .overflow-y-auto{height:500px}"
               ) as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", label="Question").style(
            container=False)

    txt.submit(get_knowledge_based_answer, [txt, state], [chatbot, state])

demo.launch(share=True, inbrowser=True)

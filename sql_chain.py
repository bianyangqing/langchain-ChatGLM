from langchain import OpenAI, SQLDatabase, SQLDatabaseChain, PromptTemplate
import os
from chatglm_llm import ChatGLM



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
result = db_chain.run(os.environ['question'])

print("final_result:{}".format(result))

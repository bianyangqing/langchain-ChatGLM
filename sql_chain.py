from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
import os
from chatglm_llm import ChatGLM

db = SQLDatabase.from_uri("sqlite:///test.db")
llm = OpenAI(temperature=0)


chatglm = ChatGLM()
chatglm.load_model(model_name_or_path="THUDM/chatglm-6b-int4")
chatglm.history_len = 3


db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
db_chain.run(os.environ['question'])

from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
import os
from chatglm_llm import ChatGLM
from knowledge_based_chatglm import llm_model_dict, LLM_MODEL, LLM_HISTORY_LEN

os.environ['OPENAI_API_KEY'] = os.environ.get('the_key_you_need')

db = SQLDatabase.from_uri("sqlite:///test.db")
llm = OpenAI(temperature=0)


chatglm = ChatGLM()
chatglm.load_model(model_name_or_path=llm_model_dict[LLM_MODEL])
chatglm.history_len = LLM_HISTORY_LEN


db_chain = SQLDatabaseChain(llm=chatglm, database=db, verbose=True)
db_chain.run(os.environ['question'])

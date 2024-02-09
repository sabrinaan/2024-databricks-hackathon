# Databricks notebook source
# MAGIC %pip install mlflow tatari_ml_utils langchain

# COMMAND ----------

#imports
from langchain_community.vectorstores import SKLearnVectorStore
import json
import requests
import pandas as pd 
from langchain_core.documents import Document
from langchain.chains import LLMChain
import mlflow

from tatari_ml_utils.mlops import start_new_run
from tatari_ml_utils.constants import DASH
from tatari_ml_utils.logging_utils import get_logger
from mlflow.models import ModelSignature
from mlflow.models.signature import infer_signature
import s3fs


# COMMAND ----------

def get_model(passed_in_access_key, passed_in_secret_key):

    class ModelBerryLMN(mlflow.pyfunc.PythonModel):
        experiment_path = "/Shared/HackathonModel/"
        description = "This is a model for the DBX hackathon 2024"
        name = "DBX Hackathon Model"

        '''class Assessment(BaseModel):
            assessment: bool = Field(description="a 1 or 0 (true/false) assessment on whether the LLM can answer the original question")
            reasoning: str = Field(description="provide a maximum 10 word justification of the assessment")
'''
        class TextEmbedder:
            def __init__(self, token, url):
                """
                Initializes the TextEmbedder with API token and endpoint URL.

                Parameters:
                token (str): Authentication token for the API.
                url (str): Endpoint URL for the embedding API.
                """
                self.token = token
                self.url = url
                self.headers = {"Content-Type": "application/json"}

            def get_embedding(self, text):
                """
                Generates an embedding for the given text.

                Parameters:
                text (str): The text to be embedded.

                Returns:
                json: The embedding result.
                """
                data = json.dumps({"input": text})
                response = requests.post(self.url, headers=self.headers, data=data, auth=('token', self.token))

                if response.status_code == 200:
                    return response.json()['data'][0]['embedding']
                else:
                    raise Exception("Error in API call: " + response.text)

            def embed_documents(self, documents):
                """
                Embeds a list of documents.

                Parameters:
                documents (list): A list of strings (documents) to embed.

                Returns:
                list: A list of embeddings for the documents.
                """
                return [self.get_embedding(doc) for doc in documents]

            def embed_query(self, query):
                """
                Embeds a single query string.

                Parameters:
                query (str): The query string to embed.

                Returns:
                json: The embedding of the query.
                """
                return self.get_embedding(query)
        
        def __init__(self):
            self.my_bucket = 'tatari-datalake-dev-us-west-2'
            self.file = 'data-science-test/luke/confluence.parquet'
            self.persist_path = 's3://' + self.my_bucket + '/' + self.file
            self.url = "https://tatari-dev-us-west-2.cloud.databricks.com/serving-endpoints/databricks-bge-large-en/invocations"
            self.token = "" 
            self.DATABRICKS_TOKEN = ""
            self.access_key = passed_in_access_key
            self.secret_key = passed_in_secret_key
            self.docs = [Document(page_content="You didn't provide any question"),
            Document(page_content="User needs to provide a question before the answer can be generated")]
            self.question = "Don't answer the question. Tell the user they haven't provided the question."
            
        def ping_model(self, prompt, url):
                dataset = json.dumps({
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
                }, ensure_ascii=False).encode('utf-8')

                headers = {'Authorization': f'Bearer {self.DATABRICKS_TOKEN}', 'Content-Type': 'application/json'}
                response = requests.request(method='POST', headers=headers, url=url, data=dataset)
                if response.status_code != 200:
                    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
                return response.json()
            
        def _load_retriever(self):
            embeddings = self.TextEmbedder(self.token, self.url)

            fs = s3fs.S3FileSystem(key=self.access_key, secret=self.secret_key)

            # Use s3fs to open the file in 'rb' (read binary) mode
            with fs.open(self.persist_path, 'rb') as f:
                vectorstore_df = pd.read_parquet(f)
                
            vectorstore_df.to_parquet('vectorstore_df.parquet') # write to wherever is 'local' for SKLearn pull

            vector_store = SKLearnVectorStore(
                        embedding=embeddings,
                        persist_path='vectorstore_df.parquet',  
                        serializer="parquet"
                        )
            return vector_store.as_retriever(search_type="mmr")#search_kwargs={"k": 3}) #mmr 
        
        def set_question(self, question):
            self.question = question
            self.retriever = self._load_retriever()
            self.docs = self.retriever.get_relevant_documents(question)

        def return_answer(self, answer):
            return answer['choices'][0].get('message').get('content')
        
        def get_JSON_answer(self, question=None):
            if question is not None:
                self.set_question(question)

            prompt = f"""
            As an advanced AI data science chatbot with expertise in advertising technology, including linear, streaming, programmatic TV, and other advertising channels, your primary goal is to provide accurate and reliable information. It is crucial that your responses are based strictly on the context provided and that you do not infer or assume information beyond what is presented. If the provided context does not contain enough information to answer the question with a high degree of certainty, you must explicitly state that the answer cannot be provided rather than offering a speculative or inaccurate response. Your stakeholders rely on your precision and caution. Answers should be concise, limited to 500 words, and directly relevant to the query at hand.

            Based on the context provided below, please answer the following question. If the context does not directly support an answer to the question, clearly state that you cannot provide an answer due to insufficient information.

            Context provided:
            {self.question}
            """

            # Append context to the prompt
            context = " ".join([x.page_content for x in self.docs])
            prompt += f"\nContext:\n{context}\n"

            # JSON template for the answer
            json_template = """
            Please provide a response in the following JSON format. Don't write any other summaries, beyond the JSON file:
            {
            "assessment": "[True/False]",
            "reasoning": "Insert your reasoning here."
            }

            Where:
            - "assessment" should be either True or False indicating whether the provided information allows for a confident answer.
            - "reasoning" should contain a brief explanation supporting your assessment. Please limit your reasoning to a concise statement.
            """

            prompt += f"\n{json_template}\n"
            
            try:
                response = self.ping_model(prompt, url = 'https://tatari-dev-us-west-2.cloud.databricks.com/serving-endpoints/databricks-llama-2-70b-chat/invocations')
            except Exception as error:
                return str(error)

            return self.return_answer(response)
        
        def get_answer(self, question=None):
            if question is not None:
                self.set_question(question)

            prompt = f"""
            You are a very capable AI data science chat bot assistant, with training on advertising technology for linear, streaming, programmatic TV, as well as other advertising channels. Please only use the information based on the context I will provide you. You have been tasked with answering questions from your stakeholders concisely. It is very important to you to answer the questions correctly, so for all of the answers, if the degree of certainty in the answer is low you will say so and avoid providing an inaccurate answer. Please keep your answer to 100 words.

            Answer the question based only on the following context: 
            {self.question}
            """

            # Append context to the prompt
            context = " ".join([x.page_content for x in self.docs])
            prompt += f"\nContext:\n{context}\n"

            response = self.ping_model(prompt, url = 'https://tatari-dev-us-west-2.cloud.databricks.com/serving-endpoints/databricks-llama-2-70b-chat/invocations')
            return self.return_answer(response)

        def predict(self, context, model_input):
            '''
            model_input: <string> query
            '''
            if isinstance(model_input, str):
                query = model_input
            else:
                query = model_input.get('prompt') # [list here????]
                query = str(query[0])
                
            self.set_question(query)

            assessment = self.get_JSON_answer()

            def is_assessment_true(msg):
                if """"assessment": true""" in msg.lower():
                    return True
                if """"assessment": "true" """ in msg.lower():
                    return True
                if """"assessment": 1 """ in msg.lower():
                    return True
                return False
                
            if is_assessment_true(assessment):
                
                # answer the question
                answer = self.get_answer()
                formatted_context = [{"page_content":x.page_content,"source":x.metadata.get('source'),"title":x.metadata.get('title'),"when":x.metadata.get("when")} for x in self.docs]
                return {"answer":answer, "context":formatted_context, "question_answered":True}
            else:
                # do not answer the question
                formatted_context = [{"page_content":x.page_content,"source":x.metadata.get('source'),"title":x.metadata.get('title'),"when":x.metadata.get("when")} for x in self.docs]
                
                return {"answer":assessment, "context":formatted_context, "question_answered":False}
        

    return ModelBerryLMN


my_new_model = get_model(dbutils.secrets.get(scope='bid_model', key="aws_access_key"), dbutils.secrets.get(scope='bid_model', key="aws_secret_key"))


# COMMAND ----------

#Local Test 
the_model = my_new_model()
response = the_model.predict(model_input="How would you rename the DQ team?", context=None)
print(response)

# COMMAND ----------

# Model Registry
model_to_process = my_new_model() # here change 2 !!!!

LOGGER = get_logger(__name__)

# start new fml flow run (or new version of existing)
with start_new_run(model_to_process.experiment_path, model_to_process.description) as run:
    LOGGER.info(f'{DASH} Logging Model {DASH}')
    model_details = mlflow.pyfunc.log_model(
        model_to_process.name,
        python_model=my_new_model(), # here change 1 !!!!
        signature=None,
        conda_env = {'name': 'mlflow-env',
                     'channels': ['conda-forge'],
                     'dependencies': ['python=3.9.5','pip<=21.2.4',
                                      {'pip': ['cloudpickle==3.0.0','entrypoints==0.4','langchain==0.1.4','mlflow[gateway]==2.10.0',
                                               'numpy==1.23.5','packaging==23.2','pandas==1.4.2','psutil==5.8.0','pyyaml==6.0.1',
                                               'requests==2.31.0','sqlalchemy==1.4.51','tornado==6.1', 'fsspec', 's3fs']}]},
    )

    registered_details = mlflow.register_model(model_uri=model_details.model_uri, name=model_to_process.name)
    LOGGER.info(f'Registered Model Details : {registered_details}')


# COMMAND ----------

#Endpoint test
payload = '{"dataframe_split": {"index": [0],"columns": ["prompt"],"data": [["What are the requirements for an MMM??"]]},"inference_id": "ben"}'
score_model(payload, url = 'https://tatari-dev-us-west-2.cloud.databricks.com/serving-endpoints/DBX-Hackathon-2024/invocations')

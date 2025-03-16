import warnings
import json
warnings.filterwarnings('ignore')
from datasets import load_dataset
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from DLAIUtils import Utils

import ast
import os
import pandas as pd

class rag_pinecone:

    def setup_pinecone(self):
        '''Recreate new pinecone index'''
        # get api key
        utils = Utils()
        PINECONE_API_KEY = utils.get_pinecone_api_key()
        pinecone = Pinecone(api_key=PINECONE_API_KEY)

        INDEX_NAME = 'dl-ai-test-pinecone-a236790'
        if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
            pinecone.delete_index(INDEX_NAME)

        pinecone.create_index(name=INDEX_NAME, dimension=1024, metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'))

        self.index = pinecone.Index(INDEX_NAME)

    def retrieve_pinecone_index(self):
        ''' Fetch existing pinecone index'''
        # get api key
        utils = Utils()
        PINECONE_API_KEY = utils.get_pinecone_api_key()
        pinecone = Pinecone(api_key=PINECONE_API_KEY)

        INDEX_NAME = 'dl-ai-test-pinecone-a236790'
        self.index = pinecone.Index(INDEX_NAME)

    def import_wiki_dataset(self):
        '''Load Wiki dataset from local filesystem and load to pinecone index'''
        max_articles_num = 500
        df = pd.read_csv('C:\\Bhuvana\\Learn\\AI\\pinecone\wiki.csv', nrows=max_articles_num)
        df.head()
        # Get embeddings
        prepped = []

        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            meta = ast.literal_eval(row['metadata'])
            #meta_json = json.loads(meta)
            #print(meta.get("text"))
            embed = self.get_embeddings([meta.get("text")])
            #print(len(embed.data[0].embedding))
            #print(embed.data[0].embedding)
            # reak
            prepped.append({'id':row['id'], 
                            # 'values':ast.literal_eval(row['values']), 
                            'values':embed.data[0].embedding, 
                            'metadata':meta})
            if len(prepped) >= 250:
                self.index.upsert(prepped)
                prepped = []
        
        self.index.describe_index_stats()

    def get_openai_client(self):
        '''Initialize OpenAI Client'''
        utils = Utils()
        OPENAI_API_KEY = utils.get_openai_api_key()
        openai_client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama'
        )
        return openai_client

    def get_embeddings(self,articles):
        '''Get embeddings for a given string'''
        openai_client_local = self.get_openai_client()
        return openai_client_local.embeddings.create(input = articles, model="mxbai-embed-large")

    def get_completions(self,prompt):
        '''Get completions for a prompt. Not all models support completions. Choosing llama3.2 which can run using ollama and requires less memory'''
        openai_client_local = self.get_openai_client()
        res = openai_client_local.completions.create(
            model="llama3.2",
            prompt=prompt,
            temperature=0,
            max_tokens=636,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        print('-' * 80)
        print(res.choices[0].text)

    def run_query(self,query):
        '''Get embeddings of given query and search related entries from pinecone index'''        
        queryList = [query]
        print(queryList)
        embed = self.get_embeddings([query])
        #print(embed.data[0])
        res = self.index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)
        text = [r['metadata']['text'] for r in res['matches']]
        #print('\n'.join(text))

    def generate_advance_result(self,query):
        '''1)Get embeddings of given query, 2) search related entries from pinecone index 3) Completions using LLM'''        
        embed = self.get_embeddings([query])
        res = self.index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)

        contexts = [
            x['metadata']['text'] for x in res['matches']
        ]

        prompt_start = (
            "Answer the question based on the context below.\n\n"+
            "Context:\n"
        )

        prompt_end = (
            f"\n\nQuestion: {query}\nAnswer:"
        )

        prompt = (
            prompt_start + "\n\n---\n\n".join(contexts) + 
            prompt_end
        )
        self.get_completions(prompt)
        # print(prompt)

    def trigger_main(self):
        ## Recreate index and load with wiki data
        # self.setup_pinecone()
        # self.import_wiki_dataset()
        # Use existing pinecone index
        self.retrieve_pinecone_index()

        query_str = "What is Mahakumbh?"
        self.run_query(query_str)
        self.generate_advance_result("write an article titled: "+query_str)
        

Object = rag_pinecone()
Object.trigger_main()
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
        # get api key
        utils = Utils()
        PINECONE_API_KEY = utils.get_pinecone_api_key()
        pinecone = Pinecone(api_key=PINECONE_API_KEY)

        utils = Utils()
        INDEX_NAME = 'dl-ai-test-pinecone-a236790'
        if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
            pinecone.delete_index(INDEX_NAME)

        pinecone.create_index(name=INDEX_NAME, dimension=1024, metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'))

        self.index = pinecone.Index(INDEX_NAME)

    def import_wiki_dataset(self):
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


    def get_embeddings(self,articles):
        utils = Utils()
        OPENAI_API_KEY = utils.get_openai_api_key()
        openai_client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama'
        )
        return openai_client.embeddings.create(input = articles, model="mxbai-embed-large")

    def run_query(self):
        query = "what is the berlin wall?"
        queryList = [query]
        print(queryList)
        embed = self.get_embeddings([query])
        print(embed.data[0])
        res = self.index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)
        text = [r['metadata']['text'] for r in res['matches']]
        print('\n'.join(text))

    def trigger_main(self):
        self.setup_pinecone()
        self.import_wiki_dataset()
       # self.run_query()

Object = rag_pinecone()
Object.trigger_main()
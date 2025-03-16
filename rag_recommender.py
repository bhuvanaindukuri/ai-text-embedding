import warnings
warnings.filterwarnings('ignore')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm, trange
from DLAIUtils import Utils

import pandas as pd
import time
import os

class rag_recommender_pinecone:

    def __init__(self):
         self.INDEX_NAME = 'dl-ai-test-pinecone-a236791'
         self.INDEX_NAME_FULL = 'dl-ai-test-pinecone-a236792'

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

    def setup_pinecone_index(self,index_name):
            '''Recreate new pinecone index'''
            pinecone = self.retrieve_pinecone()

            
            if index_name in [index.name for index in pinecone.list_indexes()]:
                pinecone.delete_index(index_name)

            pinecone.create_index(name=index_name, dimension=1024, metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1'))

            return pinecone.Index(index_name)

    def retrieve_pinecone(self):
        utils = Utils()
        PINECONE_API_KEY = utils.get_pinecone_api_key()
        return Pinecone(api_key=PINECONE_API_KEY)

    def retrieve_pinecone_index(self,index_name):
        ''' Fetch existing pinecone index'''
        pinecone = self.retrieve_pinecone()
        
        return pinecone.Index(index_name)

    def import_news_dataset(self,pinecone_index):
            '''Load News dataset from local filesystem and load to pinecone index. Loads only the titles of the articles'''
            CHUNK_SIZE=400
            TOTAL_ROWS=10000
            progress_bar = tqdm(total=TOTAL_ROWS)
            chunks = pd.read_csv('C:\\Bhuvana\\Learn\\AI\\pinecone\\all-the-news-3.csv', chunksize=CHUNK_SIZE, 
                                nrows=TOTAL_ROWS)
            chunk_num = 0
            for chunk in chunks:
                titles = chunk['title'].tolist()
                embeddings = self.get_embeddings(titles)
                prepped = [{'id':str(chunk_num*CHUNK_SIZE+i), 'values':embeddings.data[i].embedding,
                            'metadata':{'title':titles[i]},} for i in range(0,len(titles))]
                chunk_num = chunk_num + 1
                if len(prepped) >= 200:
                    pinecone_index.upsert(prepped)
                prepped = []
                progress_bar.update(len(chunk))

         
            pinecone_index.describe_index_stats()

    def embed(self,embeddings, title, prepped, embed_num,pinecone_index):
        for embedding in embeddings.data:
            prepped.append({'id':str(embed_num), 'values':embedding.embedding, 'metadata':{'title':title}})
            embed_num += 1
            if len(prepped) >= 100:
                pinecone_index.upsert(prepped)
                prepped.clear()
        return embed_num

    def import_full_article_news_dataset(self,pinecone_index):
        news_data_rows_num = 100

        embed_num = 0 #keep track of embedding number for 'id'
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, 
            chunk_overlap=20) # how to chunk each article
        prepped = []
        df = pd.read_csv('C:\\Bhuvana\\Learn\\AI\\pinecone\\all-the-news-3.csv', nrows=news_data_rows_num)
        articles_list = df['article'].tolist()
        titles_list = df['title'].tolist()

        for i in range(0, len(articles_list)):
            print(".",end="")
            art = articles_list[i]
            title = titles_list[i]
            if art is not None and isinstance(art, str):
                texts = text_splitter.split_text(art)
                embeddings = self.get_embeddings(texts)
                embed_num = self.embed(embeddings, title, prepped, embed_num,pinecone_index)
        
        pinecone_index.describe_index_stats()




    def get_recommendations(self, pinecone_index, search_term, top_k=10):
        '''Search pinecone index based on given term and return result'''
        embed = self.get_embeddings([search_term]).data[0].embedding
        res = pinecone_index.query(vector=embed, top_k=top_k, include_metadata=True)
        return res

    def trigger_main(self):
        # # Setup and load the index with only titles and get recos
        # index = self.setup_pinecone_index(self.INDEX_NAME)
        # self.import_news_dataset(index)

        index = self.retrieve_pinecone_index(self.INDEX_NAME)

        reco = self.get_recommendations(index, 'Modi')
           
        for r in reco.matches:
           print(f'{r.score} : {r.metadata["title"]}')

        #Setup index and load complete articles
        # index = self.setup_pinecone_index(self.INDEX_NAME_FULL)
        # self.import_full_article_news_dataset(index)
        print("---------")
        index_full = self.retrieve_pinecone_index(self.INDEX_NAME_FULL)

        reco = self.get_recommendations(index_full, 'Modi', top_k=100)
        seen = {}
        for r in reco.matches:
            title = r.metadata['title']
            if title not in seen:
                print(f'{r.score} : {title}')
                seen[title] = '.'


object = rag_recommender_pinecone()
object.trigger_main()
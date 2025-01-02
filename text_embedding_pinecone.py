
import warnings
warnings.filterwarnings('ignore')
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from DLAIUtils import Utils
from tqdm.auto import tqdm

class text_embedding_pinecone:
    """A class to create and populate a pinecone index with Quora dataset.Also helps with the querying of index
       Code is taken from the Deeplearning course by Pinecone.

    Attributes:
        model     The model used for converting text to embeddings.
        question  The list of questions in the dataset.
        index     The pinecone index which stores the embeddings
    """
    
    def __init__(self):
        print('text_embedding_pinecone created')

    def import_questions_dataset(self):
        dataset = load_dataset('quora', split='train[240000:290000]',trust_remote_code=True)
        print(dataset[:5])
        print('-' * 50)
        questions = []
        for record in dataset['questions']:
            questions.extend(record['text'])
        self.question = list(set(questions))
        print('\n'.join(questions[:2]))
        print('-' * 50)
        print(f'Number of questions: {len(questions)}')

    def import_transformer(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        # Testing the transformation
        query = 'which city is the most populated in the world?'
        xq = self.model.encode(query)
        print(xq.shape)

    def setup_pinecone(self):
        utils = Utils()
        PINECONE_API_KEY = utils.get_pinecone_api_key()
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        #INDEX_NAME = utils.create_dlai_index_name('dl-ai')
        INDEX_NAME = 'dl-ai-test-pinecone-a236790'

        if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
            pinecone.delete_index(INDEX_NAME)
        print(INDEX_NAME)
        pinecone.create_index(name=INDEX_NAME, 
        dimension=self.model.get_sentence_embedding_dimension(), 
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'))

        self.index = pinecone.Index(INDEX_NAME)
        print(self.index)

    def get_pincone_index(self):
        utils = Utils()
        PINECONE_API_KEY = utils.get_pinecone_api_key()
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        #INDEX_NAME = utils.create_dlai_index_name('dl-ai')
        INDEX_NAME = 'dl-ai-test-pinecone-a236790'
        self.index = pinecone.Index(INDEX_NAME)



    def upsert_embeddings_pinecone(self):
        print('upsert_embeddings_pinecone called')
        batch_size=200
        vector_limit=10000

        questions = self.question[:vector_limit]

        print('Number of questions before upload'+str(len(questions)))
        for i in tqdm(range(0, len(questions), batch_size)):
            # find end of batch
            i_end = min(i+batch_size, len(questions))
            # create IDs batch
            ids = [str(x) for x in range(i, i_end)]
            # create metadata batch
            metadatas = [{'text': text} for text in questions[i:i_end]]
            # create embeddings
            xc = self.model.encode(questions[i:i_end])
            # create records list for upsert
            records = zip(ids, xc, metadatas)
            # upsert to Pinecone
            self.index.upsert(vectors=records)
            self.index.describe_index_stats()

    # small helper function so we can repeat queries later
    def run_query(self,query):
        embedding = self.model.encode(query).tolist()
        results = self.index.query(top_k=10, vector=embedding, include_metadata=True, include_values=False)
        for result in results['matches']:
          print(f"{round(result['score'], 2)}: {result['metadata']['text']}")

    def trigger_main(self):
        # # ---- Convert dataset to embeddings and store to Pinecone index ---
        # self.import_transformer()
        # self.import_questions_dataset()
        # # Recreate Pinecone Index
        # self.setup_pinecone()
        # #Upload embeddings to the index
        # self.upsert_embeddings_pinecone()

        # --- Query pinecone index ----
        self.import_transformer()
        self.get_pincone_index()
        self.run_query('which city has the highest population in the world?')
        print('----')
        query = 'how do i make chocolate cake?'
        self.run_query(query)

Object = text_embedding_pinecone()
Object.trigger_main()
    
from sentence_transformers import SentenceTransformer, util


class SbertEmbedding:

    list_text = [
    "The Eiffel Tower in Paris was completed in 1889 and stands at 324 meters tall.",
    "Dolphins are known for their intelligence and can recognize themselves in mirror.",
    "The Great Wall of China is over 13,000 miles long and was built over several dynasties.",
    "The Amazon rainforest produces about 20% of the world’s oxygen supply."
    ]
   
    def fetch_embedding_text(self,text): 
        #Get embedding for text
        model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
        return model.encode(text)


    def fetch_embeddings(self):
        embedding_storage = dict()

        #supabase_repo = SupabaseDBConnect()
        for text in self.list_text:
            embedding_result = self.fetch_embedding_text(text)            
            embedding_storage[text] = embedding_result            

        print('Size of the result map'+str(len(embedding_storage)))
        
        return embedding_storage



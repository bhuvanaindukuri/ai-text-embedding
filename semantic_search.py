from sbert import SbertEmbedding
from SupabaseConnect import SupabaseDBConnect

class semantic_search:

    sample_query_text = "Is Iron lady the tallest structure in France?"

    def search_by_embedding(query_text):
        query_embedding = SbertEmbedding().fetch_embedding_text(query_text)
        print("Embedding of sample text"+str(query_embedding))

        supabase_client = SupabaseDBConnect().connectToMyDB()
        search_result = supabase_client.rpc("match_documents",{"query_embedding":query_embedding.tolist(),"match_threshold":0.5,"match_count":1}).execute()
        print(str(search_result))

    search_by_embedding(sample_query_text)
    
    
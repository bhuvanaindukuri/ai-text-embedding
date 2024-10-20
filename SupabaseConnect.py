import os
from supabase import create_client, Client

class SupabaseDBConnect:
    
    supabase_client = None

    def __init__(self):
        self.supabase_client = self.connectToMyDB()

    def connectToMyDB(self):
        url: str = os.environ.get("SUPABASE_URL")
        api_key: str = os.environ.get("SUPABASE_API_KEY")
        db_client: Client = create_client(url,api_key)
        print("Connecting to Supabase URL:"+url)
        return db_client

    def insertToDocument(self,index,title,text,vector_result):    
        self.supabase_client.table("documents").insert({"id": index,"title":title, "body": text, "embedding":vector_result.tolist()}).execute()
        



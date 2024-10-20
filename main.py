import os
import sys

from sbert import SbertEmbedding
from SupabaseConnect import SupabaseDBConnect

def main():
    supabase_client = SupabaseDBConnect()
    text_embeddings = SbertEmbedding().fetch_embeddings()
    pk=1
    for key in text_embeddings:
        text_str = str(key)
        title =text_str[0:20:1]
        supabase_client.insertToDocument(pk,title,text_str,text_embeddings[key])
        pk=pk+1
        print(title)

main()
    

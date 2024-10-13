import os
import sys

from sbert import SbertEmbedding
from SupabaseConnect import SupabaseDBConnect

def main():
    supabase_client = SupabaseDBConnect()
    text_embeddings = SbertEmbedding().fetch_embeddings()
    pk=1
    for key in text_embeddings:
        supabase_client.insertToDocument(pk,str(key),text_embeddings[key])
        pk=pk+1

main()
    

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

text_embedding = model.encode("How crazy is Elon Musk")

print(text_embedding)
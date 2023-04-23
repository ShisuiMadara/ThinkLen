# -*- coding: utf-8 -*-

import pandas as pd 
import cuml,cupy,cudf
import numpy 
import csv 
from tqdm.notebook import tqdm_notebook
from deep_translator import GoogleTranslator
from langdetect import detect
# from cuml.feature_extraction.text import TfidfVectorizer
# from cuml.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['NUMBAPRO_NVVM'] = '/usr/lib/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/lib/cuda/nvvm/libdevice/'
os.environ['CONDA_PREFIX'] = '/usr/lib'


# translator = GoogleTranslator(source='auto', target='en')
# path_ls = ["dataset/a.csv"]


# def translate(sentence):
#     try:
#         lang =detect(sentence)
#         if lang == 'en':
#             return sentence
#         else:
#             x = translator.translate(str(sentence))
#             return x
#     except:
#         return sentence


# def create_embedding (sentence):
#   model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
#   embeddings = model.encode(str(sentence))
#   return embeddings
#   # print(embeddings)


# def knn ():

#     KNN = 50
#     BATCH_SIZE = 256
#     NUM_BATCHES = test_embed.shape[0]//BATCH_SIZE
#     model = NearestNeighbors(n_neighbors=KNN)
#     model.fit(train_embed)
#     test_pred = []

#     start = time.time()
#     for i in range(NUM_BATCHES + 1):
#         start = i*BATCH_SIZE
#         end = (i+1)*BATCH_SIZE
        
#         distances, indices = model.kneighbors(test_embed[start:end,])
        
#         for k in range(end-start):
#             IDX = np.where(distances[k,] == np.min(distances[k,]))[0]
#             IDS = indices[k,IDX]
#             ls = train_df_cpu.iloc[cupy.asnumpy(IDS)].BROWSE_NODE_ID.values
#             test_pred.append(ls)
            
#         if(i % 50 == 0):
#             print("DONE PRODUCTS :", end)
#             print("Time : ",int(time.time()-start),"sec")

# def translate ():
#     for i,path in enumerate(path_ls):
#         df = pd.read_csv(path)
#         df["TITLE"] = df["TITLE"].progress_apply(translate)
#         df["BULLET_POINTS"] = df["BULLET_POINTS"].progress_apply(translate)
#         df["DESCRIPTION"] = df["DESCRIPTION"].progress_apply(translate)
#         df.to_csv(f"/content/drive/MyDrive/datasetb2d9982/dataset/a_out.csv",index=False)

# def embed ():
#     for i,path in enumerate(path_ls):
#         df = pd.read_csv(path)
#         df["TITLE"] = df["TITLE"].progress_apply(create_embedding)
#         df["BULLET_POINTS"] = df["BULLET_POINTS"].progress_apply(create_embedding)
#         df["DESCRIPTION"] = df["DESCRIPTION"].progress_apply(create_embedding)
#         df.to_csv(f"/content/drive/MyDrive/datasetb2d9982/dataset/a_embed_out.csv",index=False)

# def test() :
#     pred_ls = [i[0] for i in test_pred]

#     result_df = pd.DataFrame.from_dict({
#         "PRODUCT_ID" : test_df_cpu["PRODUCT_ID"],
#         "PRODUCT_LENGTH" : pred_ls,
#     })
#     result_df.to_csv("./submission.csv",index=False)
#     result_df.head()


# if __name__ == "main":

    
#     translate()

#     embed()
    

    


    



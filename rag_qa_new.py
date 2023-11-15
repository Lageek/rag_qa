# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:08:05 2023

@author: Administrator
"""

from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.llms import AzureOpenAI
from langchain.evaluation.qa import QAEvalChain
from collections import Counter
from langchain.chains.question_answering import load_qa_chain

# openai参数配置
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "https://oh-ai-openai-scu.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "6f7094c03b3e448fac43473fde475a47"

#加载openai的llm
llm = AzureOpenAI(
    deployment_name="gpt-35-turbo",
    model_name="gpt-35-turbo",
    temperature=0
)
 

loader = TextLoader('shenzhen.txt',encoding='utf-8') # 载入文件
doc = loader.load()
small_size = 50
small_overlap = 10
text_splitter = RecursiveCharacterTextSplitter(chunk_size=small_size, chunk_overlap=small_overlap)#设置文本分割器参数
docs = text_splitter.split_documents(doc)#分割文本

chunk_idx = 0 #给每一个分割段，添加一个序号，表示在原文中的序号
for d_split in docs:
    d_split.metadata['chunk_idx_name'] = chunk_idx
    chunk_idx += 1

def merge_chunk(i):#文档与前后文档融合
    if i == 0:
        return  docs[0].page_content+docs[1].page_content[small_overlap:]
    elif i == len(docs):
        return docs[i-1].page_content+docs[i].page_content[small_overlap:]
    else:
        return docs[i-1].page_content+ docs[i].page_content[small_overlap:]  + docs[i+1].page_content[small_overlap:]
        
# 设置 embedding 引擎
embeddings = OpenAIEmbeddings(openai_api_key="6f7094c03b3e448fac43473fde475a47", chunk_size = 1)

# Embed 文档，将分割的文本表示为矢量形式
docsearch = FAISS.from_documents(docs, embeddings)


#qa链接
chain = load_qa_chain(llm=llm, chain_type="map_rerank")



def qa_query(query):
    result = docsearch.as_retriever().get_relevant_documents(query)
    for doc in result:
        doc.page_content =  merge_chunk(doc.metadata['chunk_idx_name'] )
    qa_result = chain.run(input_documents=result, question=query)
    return {'query':query, 'answer':qa_result}




#测试案例
examples = [
    {
        "query": "深圳国家级专精特新“小巨人”企业总数是多少？",
        "answer": "442"
    },
    {
        "query": "万元GDP水耗分别下降多少?",
        "answer": "4.2%"
    },
    {
        "query": "九大类民生领域支出是多少？",
        "answer": "3420亿元"
    },
    {
        "query": "固废处理能力是多少?",
        "answer": "27.8万吨/日"
    },
    {
        "query": "2022经济总量是多少？",
        "answer": "3.24万亿元"
    },
    {
        "query": "市场主体总量是多少?",
        "answer": "394万户"
    }
] 

#创建评估chain
eval_chain = QAEvalChain.from_llm(llm)
predictions  = list()
for example  in examples:
    q_dict = qa_query(example['query'])
    q_dict['result']= example['answer']
    predictions.append(q_dict)
    
#生成评估结果
graded_outputs = eval_chain.evaluate(examples, predictions)


#统计评估结果
print("评估结果为：", Counter([pred['results'].split('\n\n')[0] for pred in graded_outputs]) )
#查看结果对比
for i in range(len(examples)):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + examples[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['results'].split('\n\n')[0])
    print()

# 评估结果为： Counter({' CORRECT': 6})
# Example 0:
# Question: 深圳国家级专精特新“小巨人”企业总数是多少？
# Real Answer: 442
# Predicted Answer: 442
# Predicted Grade:  CORRECT

# Example 1:
# Question: 万元GDP水耗分别下降多少?
# Real Answer: 4.2%
# Predicted Answer: 4.2%
# Predicted Grade:  CORRECT

# Example 2:
# Question: 九大类民生领域支出是多少？
# Real Answer: 3420亿元
# Predicted Answer: 3420亿元
# Predicted Grade:  CORRECT

# Example 3:
# Question: 固废处理能力是多少?
# Real Answer: 27.8万吨/日
# Predicted Answer: 27.8万吨/日
# Predicted Grade:  CORRECT

# Example 4:
# Question: 2022经济总量是多少？
# Real Answer: 3.24万亿元
# Predicted Answer: 3.24万亿元
# Predicted Grade:  CORRECT

# Example 5:
# Question: 市场主体总量是多少?
# Real Answer: 394万户
# Predicted Answer: 394万户
# Predicted Grade:  CORRECT
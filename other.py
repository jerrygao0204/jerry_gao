# 向量库内容搜索
from langchain.vectorstores import FAISS
import os

# 指定向量数据库所在的目录和文件名
vector_store_dir = "./vector_store"
index_name = "chat_history"  # 不需要包含扩展名

try:
    # 加载向量数据库，指定index_name
    vectordb = FAISS.load_local(
        folder_path=vector_store_dir,
        embeddings=None,
        index_name=index_name,  # 指定索引名称
        allow_dangerous_deserialization=True
    )
    
    # 获取向量数据库中的所有文档
    docs = vectordb.docstore._dict
    
    print(f"\n向量数据库中共有 {len(docs)} 条记录")
    print("\n文档内容预览：")
    
    for doc_id, doc in docs.items():
        print(f"\nID: {doc_id}")
        print(f"内容: {doc.page_content}")
        if doc.metadata:
            print(f"元数据: {doc.metadata}")
            
except Exception as e:
    print(f"读取向量数据库时出错: {str(e)}")
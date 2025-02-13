from pathlib import Path
from typing import List, Optional
from datetime import datetime
import logging
import shutil
import faiss
from langchain_community.vectorstores import FAISS 
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """向量数据库管理器"""
    
    def __init__(self, config: dict, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.db_config = config['vector_db']
        
        # 初始化 embedding
        self.embeddings = OpenAIEmbeddings(
            model=self.db_config['embedding_function']
        )
        
        # 设置存储路径
        self.store_dir = Path(self.db_config['store_directory'])
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / f"{self.db_config['index_name']}.faiss"
        self.docstore_path = self.store_dir / f"{self.db_config['index_name']}.pkl"
        
        # 备份目录
        if self.db_config['backup_enabled']:
            self.backup_dir = Path(self.db_config['backup_directory'])
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 操作计数器（用于自动保存）
        self._write_ops_count = 0
        
        # 初始化向量存储
        self.vector_store = self._load_or_create_store()

    def _create_index(self):
        """创建 FAISS 索引"""
        dimension = self.db_config['dimension']
        if self.db_config['distance_metric'] == 'cosine':
            index = faiss.IndexFlatIP(dimension)
        elif self.db_config['distance_metric'] == 'l2':
            index = faiss.IndexFlatL2(dimension)
        else:
            index = faiss.IndexFlatIP(dimension)  # 默认使用内积

        # 如果配置了 nlist，创建 IVF 索引
        if self.db_config.get('nlist'):
            quantizer = index
            index = faiss.IndexIVFFlat(
                quantizer, 
                dimension,
                self.db_config['nlist'],
                faiss.METRIC_INNER_PRODUCT if self.db_config['distance_metric'] == 'cosine' 
                else faiss.METRIC_L2
            )
            
        # 配置 GPU 使用
        if self.db_config['use_gpu']:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as e:
                self.logger.warning(f"GPU 初始化失败，回退到 CPU: {str(e)}")
                
        return index

    def _load_or_create_store(self) -> FAISS:
        """加载或创建新的向量存储"""
        try:
            if self.index_path.exists() and self.docstore_path.exists():
                vector_store = FAISS.load_local(
                    self.store_dir,
                    self.embeddings,
                    self.db_config['index_name'],
                    allow_dangerous_deserialization=True  # 允许危险的反序列化
                )
                if self.db_config.get('nprobe'):
                    vector_store.index.nprobe = self.db_config['nprobe']
                self.logger.info("成功加载现有向量存储")
                return vector_store
            else:
                # 创建初始文档
                initial_doc = Document(
                    page_content="Vector store initialization document",
                    metadata={
                        "source": "initialization",
                        "timestamp": datetime.now().isoformat(),
                        "version": "1.0"
                    }
                )
    
                # 不直接传入 index 参数
                vector_store = FAISS.from_documents(
                    documents=[initial_doc],
                    embedding=self.embeddings,
                    allow_dangerous_deserialization=True  # 允许危险的反序列化
                )
                # 如果需要，在创建后配置索引参数
                if self.db_config.get('nprobe'):
                    vector_store.index.nprobe = self.db_config['nprobe']
                    
                self._save_store(vector_store)
                self.logger.info("创建了新的向量存储")
                return vector_store
        except Exception as e:
            self.logger.error(f"加载/创建向量存储失败: {str(e)}")
            raise


    def _save_store(self, vector_store: Optional[FAISS] = None):
        """保存向量存储到磁盘"""
        try:
            store_to_save = vector_store or self.vector_store
            store_to_save.save_local(
                self.store_dir,
                self.db_config['index_name']
            )
            self.logger.info("向量存储已保存到磁盘")
            
            # 如果启用了备份且到达备份间隔
            if (self.db_config['backup_enabled'] and 
                self._write_ops_count % (self.db_config['backup_interval'] * 3600) == 0):
                self._create_backup()
                
        except Exception as e:
            self.logger.error(f"保存向量存储失败: {str(e)}")
            raise

    def _create_backup(self):
        """创建向量存储备份"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f"backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # 复制文件到备份目录
            shutil.copy2(self.index_path, backup_path / self.index_path.name)
            shutil.copy2(self.docstore_path, backup_path / self.docstore_path.name)
            
            # 删除旧备份
            backups = sorted(self.backup_dir.glob("backup_*"))
            if len(backups) > self.db_config['max_backups']:
                for backup in backups[:-self.db_config['max_backups']]:
                    shutil.rmtree(backup)
                    
            self.logger.info(f"创建了新的备份: {backup_path}")
        except Exception as e:
            self.logger.error(f"创建备份失败: {str(e)}")
            raise

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """添加文本到向量存储"""
        try:
            # 批量处理
            batch_size = self.db_config['batch_size']
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size] if metadatas else None
                
                self.vector_store.add_texts(batch_texts, metadatas=batch_metadatas)
                self._write_ops_count += 1
                
                # 如果启用了自动保存且达到保存间隔
                if (self.db_config['auto_save'] and 
                    self._write_ops_count % self.db_config['save_interval'] == 0):
                    self._save_store()
                    
            self.logger.info(f"成功添加 {len(texts)} 条文本到向量存储")
        except Exception as e:
            self.logger.error(f"添加文本到向量存储失败: {str(e)}")
            raise

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """执行相似性搜索"""
        try:
            k = k or self.db_config['default_top_k']
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            self.logger.error(f"相似性搜索失败: {str(e)}")
            raise
            
    def clear_store(self):
        """清空向量存储"""
        try:
            # 创建备份后再清空
            if self.db_config['backup_enabled']:
                self._create_backup()
                
            if self.index_path.exists():
                self.index_path.unlink()
            if self.docstore_path.exists():
                self.docstore_path.unlink()
                
            self.vector_store = self._load_or_create_store()
            self._write_ops_count = 0
            self.logger.info("向量存储已清空")
        except Exception as e:
            self.logger.error(f"清空向量存储失败: {str(e)}")
            raise
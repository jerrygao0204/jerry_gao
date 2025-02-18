import logging
import sys
from pathlib import Path
import yaml
from typing import List, Optional
from langchain.schema import Document
from vector_store import VectorStoreManager
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 加载配置文件
def load_config():
    try:
        config_path = Path("config.yaml")
        print(f"尝试加载配置文件: {config_path.absolute()}")
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path.absolute()}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        if config is None:
            raise ValueError("配置文件为空或格式错误")
            
        print("成功加载配置文件")
        return config
    except Exception as e:
        print(f"加载配置文件时出错: {str(e)}")
        raise

class VectorStoreService:
    def __init__(self, config: dict):
        """初始化向量存储服务"""
        self.logger = self._setup_logger()
        self.vector_store_manager = VectorStoreManager(config)

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
    
        # 清除旧的处理器
        if logger.hasHandlers():
            logger.handlers.clear()
    
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
    
        # file_handler = logging.FileHandler('./logs/vector_store.log')
        # file_handler.setLevel(logging.INFO)
    
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        # file_handler.setFormatter(formatter)
    
        logger.addHandler(console_handler)
        # logger.addHandler(file_handler)
    
        return logger


    def add_texts_to_vector_store(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """更新向量存储库"""
        try:
            # 添加文本和元数据到向量存储库
            self.vector_store_manager.add_texts(texts, metadatas=metadatas)
            self.logger.info("向量存储库已更新")
            
            # 保存更新后的向量存储库到磁盘
            self.vector_store_manager._save_store()
            self.logger.info("向量存储已保存到磁盘")
        except Exception as e:
            self.logger.error(f"更新向量存储库失败: {str(e)}")
            

    def clear_vector_store(self):
        """清空向量存储库"""
        try:
            self.vector_store_manager.clear_store()
            self.logger.info("向量存储库已清空")
        except Exception as e:
            self.logger.error(f"清空向量存储库失败: {str(e)}")
            raise

    def query_vector_store(self, query: str):
        """查询向量存储库"""
        try:
            result = self.vector_store_manager.similarity_search(query)
            if isinstance(result, Document):
                self.logger.info(f"文档内容: {result.page_content}")
                self.logger.info(f"元数据: {result.metadata}")
            else:
                self.logger.info(result)
            return result
        except Exception as e:
            self.logger.error(f"查询向量存储库失败: {str(e)}")
            raise

def main():
    try:
        # 加载配置文件
        print("正在加载配置文件...")
        config = load_config()
        print(f"加载的配置: {config}")
        
        # 创建向量存储服务
        print("正在创建向量存储服务...")
        vector_service = VectorStoreService(config)
        
        # 清空向量存储库（确保测试数据不会干扰正式数据）
        print("正在清空向量存储库...")
        vector_service.clear_vector_store()

        # 加载待分割长文本
        with open('../tests/state_of_the_union.txt', encoding='utf-8') as f:
            state_of_the_union = f.read()
        
        # 创建一个 RecursiveCharacterTextSplitter 实例
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 100,
            chunk_overlap  = 20,
            length_function = len,
            add_start_index = True,
        )
        
        # 分割文本
        docs = text_splitter.create_documents([state_of_the_union])

        # 示例文本和元数据
        texts = [doc.page_content for doc in docs]
        metadatas = [
            {
                "source": "example1",
                "timestamp": "2025-02-18"
            },
            {
                "source": "example2",
                "timestamp": "2025-02-18"
            }]
        
        
        # 更新向量存储库
        print("正在更新向量存储库...")
        vector_service.add_texts_to_vector_store(texts, metadatas)
        
        # 查询向量存储库
        print("正在查询向量存储库...")
        query = "今天星期几"
        result = vector_service.query_vector_store(query)
        
    except FileNotFoundError as e:
        print(f"找不到配置文件: {str(e)}")
    except yaml.YAMLError as e:
        print(f"配置文件格式错误: {str(e)}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        print("错误类型:", type(e))
        import traceback
        print("完整错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()



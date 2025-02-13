from typing import Optional, List
from langchain.tools import tool
from datetime import datetime
import numexpr
import requests
import logging
import yaml  # 添加这行导入语句
from pathlib import Path
from langchain_community.tools.tavily_search import TavilySearchResults
import json

logger = logging.getLogger(__name__)

def load_config():
    try:
        config_path = Path("config.yaml")
        # print(f"尝试加载配置文件: {config_path.absolute()}")
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: tool_{config_path.absolute()}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        if config is None:
            raise ValueError("配置文件为空或格式错误")
            
        # print("成功加载配置文件")
        return config
    except Exception as e:
        # print(f"加载配置文件时出错: {str(e)}")
        raise

config = load_config()

class Tools:
    def __init__(self, config: dict, vector_store_manager=None, logger=None):
        self.config = config
        if vector_store_manager is None:
            raise ValueError("vector_store_manager 不能为空")
        self.vector_store_manager = vector_store_manager
        self.logger = logger or logging.getLogger(__name__)

    @tool("calculator")
    def calculator(query: str) -> str:
        """进行基础数学计算"""
        try:
            # 验证输入是否为合法的数学表达式
            query = query.strip()
            logger.info(f"当前消息总query数: {query}")
            if not query:
                return "错误：输入为空"
            # 可以添加更多验证逻辑
            result = numexpr.evaluate(query)
            return str(float(result))
        except Exception as e:
            return f"计算错误: {str(e)}"

    @tool("weather")
    def weather(location: str) -> str:
        """获取指定地点的天气信息"""
        API_KEY = config['tools']['weather'].get('api_key')
        if not API_KEY:
            return "错误: 未配置天气API密钥"
            
        try:
            url = f"{config['tools']['weather']['base_url']}/current.json"
            params = {
                "key": API_KEY,
                "q": location
            }
            response = requests.get(
                url, 
                params=params,
                timeout=config['tools']['weather']['timeout']
            )
            response.raise_for_status()
            
            data = response.json()
            temp = data['current']['temp_c']
            condition = data['current']['condition']['text']
            
            return f"{location}当前温度{temp}°C, {condition}"
        except requests.RequestException as e:
            self.logger.error(f"天气查询失败: {str(e)}")
            return f"天气查询失败: {str(e)}"

    @tool("search_vector_db")
    def search_vector_db(self, query: str, k: int = 3, lambda_mult: float = 0.9) -> str:
        """搜索向量数据库中的相似内容"""
        if not self.vector_store_manager:
            return "错误: 向量存储管理器未初始化"
        
        try:
            results = self.vector_store_manager.similarity_search(query, k=k, lambda_mult=lambda_mult)
            if not results:
                return "未找到相关内容"
            
            response_list = [f"{i + 1}. {doc.page_content}{' 元数据: ' + str(doc.metadata) if doc.metadata else ''}" for i, doc in enumerate(results)]
            response = "找到以下相关内容：\n" + "\n".join(response_list)
            
            return response
        except AttributeError as e:
            self.logger.error(f"向量数据库搜索失败: {str(e)}")
            return f"搜索失败: {str(e)}"
        except Exception as e:
            self.logger.error(f"未知错误: {str(e)}")
            return f"搜索失败: {str(e)}"

    
    def process_tool_call(self, tool_call: dict) -> dict:
        """处理工具调用并返回响应"""
        try:
            if not isinstance(tool_call, dict):
                raise TypeError("tool_call 必须是字典类型")
                
            if 'function' not in tool_call or 'name' not in tool_call['function']:
                raise ValueError("无效的工具调用格式")
                
            tool_name = tool_call['function']['name']
            tool_args = json.loads(tool_call['function']['arguments'])
            tool_method = getattr(self, tool_name, None)
            if not tool_method:
                raise ValueError(f"未找到工具: {tool_name}")
            
            # 根据工具名称分别处理参数
            if tool_name == 'calculator':
                query = tool_args.get('query', '')
                # query = tool_args
                if not query:
                    raise ValueError("工具调用中缺少 'query' 参数")
                result = tool_method.invoke(str(query))
            elif tool_name == 'weather':
                result = tool_method.invoke(tool_args.get('location', ''))
            elif tool_name == 'search_vector_db':
                result = tool_method.invoke(
                    query=tool_args.get('query', ''),
                    k=tool_args.get('k', 3),
                    lambda_mult=tool_args.get('lambda_mult', 0.9)
                )
            else:
                raise ValueError(f"未知的工具类型: {tool_name}")
            logger.info(f"当前消息总tool_call——id数: {tool_call['id']}")
            logger.info(f"当前消息总output数: {str(result)}")
            return {
                'tool_call_id': tool_call['id'],
                'output': str(result),
                'status': 'success'
            }
        except Exception as e:
            self.logger.error(f"工具调用失败: {str(e)}", exc_info=True)
            return {
                'tool_call_id': tool_call.get('id', 'unknown'),
                'output': f"工具调用失败: {str(e)}",
                'status': 'error'
            }


    @property
    def tool_list(self):
        return [
            self.calculator,
            self.weather,
            self.search_vector_db
            # self.process_tool_call 
        ]

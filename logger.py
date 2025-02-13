import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os
from datetime import datetime

class LoggerManager:
    """日志管理类"""
    
    def __init__(self, config: dict):
        """
        初始化日志管理器
        
        Args:
            config: 配置字典，需包含logging相关配置
        """
        self.config = config['logging']
        self.log_dir = Path(self.config.get('directory', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志文件名
        self.log_filename = self._get_log_filename()
        
        # 初始化日志配置
        self._setup_logging()
        
        # 获取logger实例
        self.logger = logging.getLogger(self.config.get('name', __name__))

    def _get_log_filename(self) -> str:
        """生成日志文件名"""
        filename_pattern = self.config.get('filename_pattern', 'app_{date}.log')
        date_str = datetime.now().strftime('%Y%m%d')
        return filename_pattern.format(date=date_str)

    def _setup_logging(self):
        """配置日志系统"""
        log_level = getattr(logging, self.config.get('level', 'INFO'))
        log_format = self.config.get('format', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 创建格式化器
        formatter = logging.Formatter(log_format)
        
        # 根据配置创建处理器
        handlers = []
        
        # 文件处理器
        if self.config.get('file_logging', True):
            file_handler = self._create_file_handler(formatter)
            handlers.append(file_handler)
        
        # 控制台处理器
        if self.config.get('console_logging', True):
            console_handler = self._create_console_handler(formatter)
            handlers.append(console_handler)
        
        # 配置根日志记录器
        logging.basicConfig(
            level=log_level,
            handlers=handlers
        )

    def _create_file_handler(self, formatter):
        """创建文件处理器"""
        log_path = self.log_dir / self.log_filename
        
        # 根据配置选择处理器类型
        if self.config.get('rotate_by_size', False):
            handler = RotatingFileHandler(
                log_path,
                maxBytes=self.config.get('max_bytes', 10*1024*1024),  # 默认10MB
                backupCount=self.config.get('backup_count', 5)
            )
        else:
            handler = TimedRotatingFileHandler(
                log_path,
                when=self.config.get('rotate_when', 'midnight'),
                interval=self.config.get('rotate_interval', 1),
                backupCount=self.config.get('backup_count', 7)
            )
        
        handler.setFormatter(formatter)
        return handler

    def _create_console_handler(self, formatter):
        """创建控制台处理器"""
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        return handler

    def get_logger(self):
        """获取logger实例"""
        return self.logger
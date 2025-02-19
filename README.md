# 大模型（LLMs）应用对话系统项目介绍

<p align="center">
    <br> <a href="README-en.md">English</a> | 中文
</p>


这是一个基于LangChain和OpenAI构建的智能对话系统，提供了一个集成向量数据库检索、工具调用、上下文管理等功能的完整解决方案。系统采用模块化设计，具有良好的扩展性和可维护性。

## 特性

- **状态图管理**：基于LangGraph实现对话流程的可视化和控制

- **智能上下文**：自动管理对话历史，支持多种消息裁剪策略

- **知识检索**：集成向量数据库，支持相似内容搜索

- **工具系统**：可扩展的工具调用框架

- **配置管理**：基于YAML的灵活配置系统

- **日志系统**：完善的日志记录和异常处理机制

- **Token控制**：智能的token计数和管理

- **状态追踪**：完整的工具调用状态管理

## 拉取代码

你可以通过克隆此仓库到你的本地机器来开始：

```shell
git clone https://github.com/jerrygao0204/jerry_gao.git
```

然后导航至目录，并按照单个模块的指示开始操作。

## 功能模块

本项目使用 Python v3.11 开发，完整 Python 依赖软件包见[requirements.txt](requirements.txt)。

关键依赖的官方文档如下：

- Python 环境管理 [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- Python 交互式开发环境 [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- 大模型应用开发框架 [LangChain](https://python.langchain.com/docs/get_started/installation),[LangSmith](https://smith.langchain.com/onboarding?organizationId=3e596660-0057-4cc6-b49a-bec8550f5d17&step=1),[LangGraph](https://langchain-ai.github.io/langgraph/)
- [OpenAI Python SDK ](https://langchain-ai.github.io/langgraph/) 

### 对话管理

- **对话历史维护和上下文管理**
- **支持总结模式的智能消息裁剪**
- **自动消息去重和规范化处理**
- **灵活的会话状态控制**

### 知识库系统

- **相似度搜索功能**
- **搜索结果格式化输出**
- **向量数据库接入和管理（待优化）
- **支持元数据管理和检索（待优化）

### 工具框架

- **自定义工具注册机制****
- **工具调用状态跟踪**
- **工具响应配对系统**
- **异步调用支持（待开发）

### 系统配置

- **模型参数配置**
- **日志级别设置**
- **向量库参数管理**
- **系统行为控制**

### 异常处理

- **多层级错误捕获**
- **详细日志记录**
- **用户友好的错误提示**
- **故障恢复机制（待开发）

**以下是详细的安装指导（以 Ubuntu 操作系统为例）**：

### 安装 Miniconda

```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

安装完成后，建议新建一个 Python 虚拟环境，命名为 `langchain`。

```shell
conda create -n langchain python=3.10

# 激活环境
conda activate langchain 
```

之后每次使用需要激活此环境。


### 安装 Python 依赖软件包

```shell
pip install -r requirements.txt
```

### 配置 OpenAI API Key

根据你使用的命令行工具，在 `~/.bashrc` 或 `~/.zshrc` 中配置 `OPENAI_API_KEY` 环境变量：

```shell
export OPENAI_API_KEY="xxxx"
```

### 安装和配置 Jupyter Lab

上述开发环境安装完成后，使用 Miniconda 安装 Jupyter Lab：

```shell
conda install -c conda-forge jupyterlab
```

使用 Jupyter Lab 开发的最佳实践是后台常驻，下面是相关配置（以 root 用户为例）：

```shell
# 生成 Jupyter Lab 配置文件，
jupyter lab --generate-config
```

打开上面执行输出的`jupyter_lab_config.py`配置文件后，修改以下配置项：

```python
c.ServerApp.allow_root = True # 非 root 用户启动，无需修改
c.ServerApp.ip = '*'
```

使用 nohup 后台启动 Jupyter Lab
```shell
$ nohup jupyter lab --port=8000 --NotebookApp.token='替换为你的密码' --notebook-dir=./ &
```

Jupyter Lab 输出的日志将会保存在 `nohup.out` 文件（已在 .gitignore中过滤）。

## 技术架构

- **框架: LangChain/LangGraph**
- **模型: OpenAI ChatGPT**：后续会增加其他模型的调用
- **Token处理: tiktoken**
- **配置管理: PyYAML**
- **日志系统: Python logging**
- **存储: Vector Store**

## 部署要求

- **Python 3.x**
- **OpenAI API密钥**
- **向量数据库支持**
- **YAML配置文件**

## 使用场景

- **智能客服系统**
- **知识库问答**
- **工具型助手**
- **对话式应用**
- **智能流程自动化**

## 待实现/优化功能

### 待优化
- **向量数据库接入和管理**
  
      已完成向量库维护专用模块（2025.2.18）
- **支持元数据管理和检索**

      已支持元数据配置及过滤功能（2025.2.19）
  
### 待实现
- **故障恢复机制**
- **小模型的意图分类**
- **网络查询功能**
- **增加可选大模型**
- **大模型自动选择**
- **prompt模板自动生成**
- **反思机制**

## 贡献

贡献是使开源社区成为学习、激励和创造的惊人之处。非常感谢你所做的任何贡献。如果你有任何建议或功能请求，请先开启一个议题讨论你想要改变的内容。

<a href='https://github.com/repo-reviews/repo-reviews.github.io/blob/main/create.md' target="_blank"><img alt='Github' src='https://img.shields.io/badge/review_me-100000?style=flat&logo=Github&logoColor=white&labelColor=888888&color=555555'/></a>

## 许可证

该项目根据Apache-2.0许可证的条款进行许可。详情请参见[LICENSE](LICENSE)文件。

## 联系

Jerry gao - jerry.gao0204@@gmail.com
项目链接：https://github.com/jerrygao0204/jerry_gao/

项目链接: 

## 特别感谢

特别感谢彭老师以及Lewis老师的指导。

推荐彭老师的项目，项目链接: https://github.com/DjangoPeng/openai-quickstart
## ⭐️⭐️

<a href="https://star-history.com/#DjangoPeng/openai-quickstart&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=DjangoPeng/openai-quickstart&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=DjangoPeng/openai-quickstart&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=DjangoPeng/openai-quickstart&type=Date" />
  </picture>
</a>

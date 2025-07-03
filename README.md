# 大语言模型部署
本项目基于阿里云 ModelScope 魔搭社区平台实现，依托平台提供的开源中文大语言模型资源，包括通义千问 Qwen-7B-Chat 和智谱 ChatGLM3-6B，结合本地部署环境，完成模型的下载、部署与推理测试实践。

## 部署流程
### 1. 打开终端命令行环境：
* 安装基础依赖（兼容transformers 4.33.3 和neuralchat）
```
pip install \
"intel-extension-for-transformers==1.4.2" \
"neural-compressor==2.5" \
"transformers==4.33.3" \
"modelscope==1.9.5" \
"pydantic==1.10.13" \
"sentencepiece" \
"tiktoken" \
"einops" \
"transformers_stream_generator" \
"uvicorn" \
"fastapi" \
"yacs" \
"setuptools_scm"
```
* 安装fschat（需要启用PEP517 构建）：
```
pip install fschat --use-pep517
```
### 2. 下载大模型到本地
* 切换到数据目录
```
cd /mnt/data
```
* 下载对应大模型
```
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git  # 智谱ChatGLM3-6B 
git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git  # 通义千问Qwen-7B-Chat
```
### 3. 构建实例（以通义千问为例）
* 切换到工作目录
```
cd /mnt/workspace
```
* 编辑推理脚本 test_qwen.py：
```
touch test_qwen.py # 建立文件
vim test_qwen.py # 使用vim编辑
```
* 进入编辑模式，编写Python代码：
```
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
model_name = "/mnt/data/Qwen-7B-Chat" # 本地路径
prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少2、夏天：能穿多少穿多少"
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  trust_remote_code=True,
  torch_dtype="auto" # 自动选择float32/float16（根据模型配置）
).eval()
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```
* 写完按Esc，输入:wq，回车，保存退出
* 运行实例
```
python test_qwen.py
```

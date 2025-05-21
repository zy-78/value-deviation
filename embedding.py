import os
import json
import pickle
import argparse
import asyncio
import time
from tqdm import tqdm
import numpy as np
import tiktoken
from openai import OpenAI

"""
脚本: embedding.py
功能: 从JSON文件生成嵌入向量并保存为PKL文件，用于OPO系统
使用方法: 
    python embedding.py --input xxxxxx.json --output rules/morality_embed_text_pairs.pkl --api_key YOUR_API_KEY --api_base YOUR_API_BASE
"""

# 默认API端点
DEFAULT_API_BASE = "https://api.openai.com/v1"  # OpenAI官方API
DEFAULT_API_BASE =    #自己的url


# 分块函数，将列表分成指定大小的块
def chunks(lst, n):
    """将列表分成n大小的块"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# 截断文本令牌，确保不超过最大令牌数
def truncate_text_tokens(texts, encoding_name='cl100k_base', max_tokens=8191):
    """将文本截断到指定的最大令牌数"""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        truncated_texts = []
        for text in texts:
            tokens = encoding.encode(text)
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                truncated_text = encoding.decode(truncated_tokens)
                truncated_texts.append(truncated_text)
            else:
                truncated_texts.append(text)
        return truncated_texts
    except Exception as e:
        print(f"令牌截断失败，将使用字符截断: {e}")
        # 如果tiktoken不可用，简单截断字符
        return [text[:max_tokens * 4] for text in texts]  # 粗略估计每个token约4个字符

class EmbeddingGenerator:
    def __init__(self, api_key, api_base=DEFAULT_API_BASE, model="text-embedding-ada-002"):
        """
        初始化嵌入生成器
        
        参数:
            api_key: OpenAI API密钥
            api_base: API基础URL
            model: 使用的嵌入模型
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        
    async def get_embedding_async(self, text):
        """异步获取单个文本的嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            # 返回一个0向量作为回退
            return [0.0] * 1536  # OpenAI嵌入通常是1536维
            
    async def get_embeddings_batch(self, texts):
        """异步获取一批文本的嵌入向量"""
        tasks = [self.get_embedding_async(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    def generate_embeddings(self, texts, batch_size=5):
        """为一系列文本生成嵌入向量"""
        all_embeddings = []
        batches = list(chunks(texts, batch_size))
        
        for batch in tqdm(batches, desc="生成嵌入向量"):
            try:
                # 限制文本长度
                truncated_batch = truncate_text_tokens(batch, max_tokens=8191)
                # 异步获取该批次的嵌入
                embeddings = asyncio.run(self.get_embeddings_batch(truncated_batch))
                all_embeddings.extend(embeddings)
                # 避免API速率限制
                time.sleep(0.5)
            except Exception as e:
                print(f"处理批次时出错: {e}")
                # 对于失败的批次，添加0向量
                all_embeddings.extend([[0.0] * 1536] * len(batch))
        
        return all_embeddings

def process_json_rules(json_file, output_file, api_key, api_base=DEFAULT_API_BASE, 
                      batch_size=5, field_name="text", id_field=None):
    """
    处理JSON文件中的规则，生成嵌入向量并保存为PKL文件
    
    参数:
        json_file: 输入的JSON文件路径
        output_file: 输出的PKL文件路径
        api_key: OpenAI API密钥
        api_base: API基础URL
        batch_size: 批处理大小
        field_name: JSON中包含规则文本的字段名
        id_field: JSON中包含规则ID的字段名（可选）
    """
    # 加载JSON文件
    print(f"正在加载 {json_file}...")
    
    # 检测文件是否是JSONL格式（每行一个JSON对象）
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # 回到文件开头
            f.seek(0)
            
            rules = []
            # 检查是否是JSONL格式
            if first_line.startswith('{') and first_line.endswith('}'):
                try:
                    # 尝试解析为JSONL
                    for line in f:
                        if line.strip():  # 跳过空行
                            rules.append(json.loads(line))
                    print(f"检测到JSONL格式文件（每行一个JSON对象）")
                except json.JSONDecodeError:
                    # 回到文件开头，尝试作为单个JSON解析
                    f.seek(0)
                    data = json.load(f)
                    
                    # 检查数据格式
                    if isinstance(data, dict) and "rules" in data:
                        # 格式 {"rules": [...]}
                        rules = data["rules"]
                    elif isinstance(data, list):
                        # 格式 [rule1, rule2, ...]
                        rules = data
                    else:
                        print("不支持的JSON格式。应为规则列表或包含'rules'字段的字典。")
                        return
            else:
                # 常规JSON格式
                data = json.load(f)
                
                # 检查数据格式
                if isinstance(data, dict) and "rules" in data:
                    # 格式 {"rules": [...]}
                    rules = data["rules"]
                elif isinstance(data, list):
                    # 格式 [rule1, rule2, ...]
                    rules = data
                else:
                    print("不支持的JSON格式。应为规则列表或包含'rules'字段的字典。")
                    return
    except Exception as e:
        print(f"无法加载JSON文件: {e}")
        return
    
    print(f"加载了 {len(rules)} 条规则")
    
    # 提取规则文本
    texts = []
    for rule in rules:
        if isinstance(rule, str):
            texts.append(rule)
        elif isinstance(rule, dict) and field_name in rule:
            texts.append(rule[field_name])
        else:
            print(f"警告: 跳过无效规则 {rule}")
    
    print(f"提取了 {len(texts)} 条有效文本")
    
    # 初始化嵌入生成器
    generator = EmbeddingGenerator(api_key, api_base)
    
    # 生成嵌入
    embeddings = generator.generate_embeddings(texts, batch_size)
    
    # 创建文本-嵌入对
    text_embed_pairs = list(zip(texts, embeddings))
    
    # 保存到PKL文件
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(text_embed_pairs, f)
    
    print(f"成功保存 {len(text_embed_pairs)} 对文本-嵌入向量到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description="从JSON文件生成嵌入向量并保存为PKL文件")
    parser.add_argument("--input", type=str,  help="输入的JSON文件路径",default=r"rule.jsonl")
    parser.add_argument("--output", type=str,  help="输出的PKL文件路径",default=r"rules\morality_embed_text_pairs.pkl")
    parser.add_argument("--api_key", type=str, help="OpenAI API密钥",required=True)
    parser.add_argument("--api_base", type=str, default=DEFAULT_API_BASE, help="API基础URL")
    parser.add_argument("--batch_size", type=int, default=5, help="批处理大小")
    parser.add_argument("--field_name", type=str, default="text", help="JSON中包含规则文本的字段名")
    parser.add_argument("--id_field", type=str, default=None, help="JSON中包含规则ID的字段名（可选）")
    
    args = parser.parse_args()
    
    # 如果未提供API密钥，尝试从环境变量获取
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("请输入OpenAI API密钥: ")
    
    process_json_rules(args.input, args.output, api_key, args.api_base, 
                      args.batch_size, args.field_name, args.id_field)

if __name__ == "__main__":
    main()
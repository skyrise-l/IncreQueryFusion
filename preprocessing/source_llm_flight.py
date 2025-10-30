import os
import pandas as pd
import numpy as np
from utils.result_judge import ResultJudge
import json
import re
import time

def extract_source_name(filename):
    """提取文件名的第一部分作为源名称"""
    base_name = os.path.splitext(filename)[0]
    source_name = re.split(r'[_\d]', base_name)[0]
    return source_name

def analyze_data_quality(file_path):
    """分析单个文件的数据质量"""
    try:
        df = pd.read_csv(file_path, sep='\t', dtype=str)
        
        if 'Source' in df.columns:
            df = df.drop('Source', axis=1)
        
        # 统一定义分析的列
        columns_to_analyze = ['Author', 'Title']
        
        filename = os.path.basename(file_path)
        source_name = extract_source_name(filename)
        
        quality_info = {
            'file_name': filename,
            'source_name': source_name,
            'total_records': len(df),
            'columns_analysis': {},
            'sample_data': {}
        }
        
        for col in columns_to_analyze:
            if col not in df.columns:
                quality_info['columns_analysis'][col] = {
                    'exists': False,
                    'nan_ratio': 1.0,
                    'non_nan_count': 0
                }
                continue
                
            # 修复空值处理
            nan_mask = df[col].isna()
            # 处理字符串类型的空值
            if df[col].dtype == 'object':
                nan_mask = nan_mask | (df[col] == '') | (df[col].str.lower() == 'nan')
            
            nan_ratio = nan_mask.mean()
            non_nan_count = len(df) - nan_mask.sum()
            
            quality_info['columns_analysis'][col] = {
                'exists': True,
                'nan_ratio': round(nan_ratio, 4),
                'non_nan_count': int(non_nan_count),
                'is_all_nan': nan_ratio == 1.0
            }
            
            non_nan_values = df[col][~nan_mask].head(3).tolist()
            quality_info['sample_data'][col] = non_nan_values
        
        return quality_info
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return None

def generate_batch_llm_prompt(batch_quality_infos, batch_size=5):
    """为一批数据源生成LLM提示词"""
    
    batch_summary = []
    
    for i, quality_info in enumerate(batch_quality_infos):
        source_name = quality_info['source_name']
        
        source_summary = f"\n--- 数据源 {i+1}: {source_name} ---\n"
        source_summary += f"总记录数: {quality_info['total_records']}\n"
        
        # 使用正确的列名
        key_columns = ['Author', 'Title']
        missing_info = []
        
        for col in key_columns:
            if col in quality_info['columns_analysis']:
                analysis = quality_info['columns_analysis'][col]
                if analysis['exists'] and not analysis['is_all_nan']:
                    missing_info.append(f"{col}:{analysis['nan_ratio']*100:.1f}%缺失")
        
        source_summary += "缺失情况: " + "; ".join(missing_info) + "\n"
        
        batch_summary.append(source_summary)
    
    example_sources = [info['source_name'] for info in batch_quality_infos]
    example_json = "{" + ", ".join([f'"{source}": 等级' for source in example_sources]) + "}"
    
    prompt = f"""
## 背景信息
这是多个股票/金融文献数据源的质量评估任务。数据源包含作者、标题等学术文献信息。

## 需要评估的数据源列表
{''.join(batch_summary)}

## 评估要求
请基于数据质量信息和你的对金融文献数据源的领域知识了解（比如有些知名学术机构、出版社更可信等），为每个数据源给出初始可信度分级（只需数字等级，不要理由）：
- 1: 第一级信任（数据质量很高，可信度最高）
- 2: 第二级信任（数据质量良好，可信度较高）  
- 3: 第三级信任（数据质量一般，可信度中等）
- 4: 第四级信任（数据质量较差，可信度较低）
- 5: 第五级信任（数据质量很差，可信度最低）

请严格按照以下格式返回，只包含JSON字典，不要其他内容：
{example_json}

请确保使用源名称作为键（而不是完整文件名）。

注意的评估的逻辑是，首先解析名字，思考这是来源于哪个官方机构、学术出版社或研究机构，这占据评估的70%，此外，关键属性的缺失比例，和数据量（数据量特别小的降低可靠）也是考虑的一部分。
"""
    return prompt

# 其他函数保持不变，但确保使用正确的列名
def save_results_to_file(results, output_file='data_source_confidence.json'):
    """将结果保存到文件"""
    simplified_results = []
    for result in results:
        simplified = {
            'file_name': result['file_name'],
            'source_name': result['source_name'],
            'confidence_level': result['confidence_level'],
            'total_records': result['quality_info']['total_records'],
            'key_columns_missing_ratio': {}
        }
        
        # 使用正确的列名
        key_columns = ['Author', 'Title']
        for col in key_columns:
            if col in result['quality_info']['columns_analysis']:
                analysis = result['quality_info']['columns_analysis'][col]
                if analysis['exists'] and not analysis['is_all_nan']:
                    simplified['key_columns_missing_ratio'][col] = analysis['nan_ratio']
        
        simplified_results.append(simplified)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")


def assess_data_sources_batch(directory_path, batch_size=5):
    """批量评估数据源的可信度"""
    
    # 初始化LLM判断器
    resultJudge = ResultJudge("deepseek-api")
    
    # 收集所有文件的质量信息
    all_quality_infos = []
    valid_files = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            print(f"正在分析: {filename}")
            
            quality_info = analyze_data_quality(file_path)
            if quality_info is not None:
                all_quality_infos.append(quality_info)
                valid_files.append(filename)
    
    if not all_quality_infos:
        print("未找到有效的数据文件")
        return []
    
    results = []
    
    # 分批处理
    for i in range(0, len(all_quality_infos), batch_size):
        batch = all_quality_infos[i:i+batch_size]
        batch_files = valid_files[i:i+batch_size]
        
        # 提取这批文件的源名称
        batch_source_names = [info['source_name'] for info in batch]
        print(f"\n正在评估批次 {i//batch_size + 1}: {', '.join(batch_source_names)}")
        
        # 生成批量提示词
        prompt = generate_batch_llm_prompt(batch, batch_size)

        print(prompt)
        
        try:
            # 调用LLM进行评估
            llm_response = resultJudge.judge(prompt)
            
            print(f"LLM响应: {llm_response}")

            print(llm_response)
            
            # 解析LLM响应（使用源名称进行匹配）
            confidence_dict = parse_llm_response(llm_response, batch)
            
            # 将结果与质量信息关联
            for quality_info in batch:
                source_name = quality_info['source_name']
                confidence_level = confidence_dict.get(source_name, 3)  # 默认3级
                
                result = {
                    'file_name': quality_info['file_name'],
                    'source_name': source_name,
                    'quality_info': quality_info,
                    'confidence_level': confidence_level
                }
                results.append(result)
                print(f"评估完成: {source_name} -> 等级{confidence_level}")
            
            # 添加延迟，避免API限制
            time.sleep(1)
            
        except Exception as e:
            print(f"批次评估失败: {str(e)}")
            # 如果批量评估失败，可以尝试逐个评估
            for quality_info in batch:
                try:
                    single_prompt = generate_single_llm_prompt(quality_info)
                    llm_response = resultJudge.judge(single_prompt)
                    confidence_level = extract_single_confidence(llm_response)
                    
                    result = {
                        'file_name': quality_info['file_name'],
                        'source_name': quality_info['source_name'],
                        'quality_info': quality_info,
                        'confidence_level': confidence_level
                    }
                    results.append(result)
                    print(f"单个评估完成: {quality_info['source_name']} -> 等级{confidence_level}")
                    
                    time.sleep(1)  # 单个评估也添加延迟
                    
                except Exception as e2:
                    print(f"单个评估也失败 {quality_info['source_name']}: {str(e2)}")
                    # 给予默认等级
                    result = {
                        'file_name': quality_info['file_name'],
                        'source_name': quality_info['source_name'],
                        'quality_info': quality_info,
                        'confidence_level': 3  # 默认中等
                    }
                    results.append(result)
    
    return results

def parse_llm_response(llm_response, batch_quality_infos):
    """解析LLM响应，提取JSON格式的置信度字典，使用源名称匹配"""
    try:
        # 尝试直接解析JSON
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            response_dict = json.loads(json_str)
        else:
            # 如果没有找到JSON，尝试从文本中提取键值对
            response_dict = {}
            lines = llm_response.split('\n')
            for line in lines:
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        key = parts[0].strip().replace("'", "").replace('"', "")
                        value_str = parts[1].strip()
                        # 提取数字
                        level_match = re.search(r'\d', value_str)
                        if level_match:
                            response_dict[key] = int(level_match.group())
        
        # 创建置信度字典，使用源名称作为键
        confidence_dict = {}
        batch_source_names = [info['source_name'] for info in batch_quality_infos]
        
        for source_name in batch_source_names:
            # 尝试多种匹配方式
            if source_name in response_dict:
                confidence_dict[source_name] = response_dict[source_name]
            else:
                # 尝试部分匹配
                matched = False
                for key in response_dict.keys():
                    if key in source_name or source_name in key:
                        confidence_dict[source_name] = response_dict[key]
                        matched = True
                        break
                
                if not matched:
                    # 如果没有匹配到，给予默认等级
                    print(f"警告: 无法为源 {source_name} 找到匹配的键。响应字典中的键: {list(response_dict.keys())}")
                    confidence_dict[source_name] = 3  # 默认中等
        
        return confidence_dict
    except Exception as e:
        print(f"解析LLM响应失败: {str(e)}")
        # 返回一个默认字典，所有源都为3级
        batch_source_names = [info['source_name'] for info in batch_quality_infos]
        return {source_name: 3 for source_name in batch_source_names}

def generate_single_llm_prompt(quality_info):
    """生成单个数据源的简化提示词（备用）"""
    prompt = f"""
数据源: {quality_info['source_name']}
记录数: {quality_info['total_records']}
请给出可信度等级(1-5): 
"""
    return prompt

def extract_single_confidence(llm_response):
    """从单个评估响应中提取等级"""
    level_match = re.search(r'[1-5]', llm_response)
    return int(level_match.group()) if level_match else 3

def main():
    """主函数"""
    directory_path = "/home/lwh/QueryFusion/data/dataset/stock/raw_data"
    
    if not os.path.exists(directory_path):
        print("目录不存在!")
        return
    
    batch_size = int(input("请输入每批评估的文件数量(推荐5-10): ").strip() or "5")
    
    print("开始批量分析数据源质量...")
    results = assess_data_sources_batch(directory_path, batch_size)
    
    if results:
        save_results_to_file(results)
        
        # 打印汇总信息
        print("\n=== 数据源可信度汇总 ===")
        for result in results:
            print(f"{result['source_name']} ({result['file_name']}): 等级{result['confidence_level']}")
            
        # 统计各级别数量
        from collections import Counter
        level_counts = Counter([r['confidence_level'] for r in results])
        print(f"\n分级统计: {dict(level_counts)}")
    else:
        print("未找到有效的数据文件或分析失败。")

if __name__ == "__main__":
    main()
import re
import json
from utils.my_api import MyApi
from utils.model_loader import EmbeddingModel
from llm.llm_server import ChatBot
import numpy as np

class QuerySourceManager:
    def __init__(self, query_folder, chain_modal = "Deepseek"):
        self.query_folder = query_folder
        self.api_key = "sk-8Pr2EG7j5AUi3UuFQL8ZgShMqQoan39uWg8qVkD6Ysssv0ow"
        self.chain_modal = chain_modal
        if chain_modal == "Deepseek":
            self.modal = ChatBot()

    def query_reasoning_chain(self, query):
        prompt = f'''
        I'm solving a QA task and would like you to parse the question and provide a reasoning chain. Each step in the reasoning chain should consist of exactly two attribute-value pairs, with implicit (unknown) attributes or values left empty ("").

    Example:

    Question: "Who are the directors of Myson?"

    Reasoning chain:
    {{
        "Step1": {{ 
            "attribute1": "",
            "value1": "Myson", 
            "attribute2": "directors",
            "value2": "Result"
        }}
    }}
    This example means we start from "Myson" (attribute unknown), deduce the value of the attribute "directors," and output that as the result.

    When multi-step reasoning is required, use "IntermediateResultX" as either "value2" or "attribute2". You can subsequently reference this "IntermediateResultX" as "value1" or "attribute1" in later steps.

    Here is the query: {query}.
    Please directly output a reasoning chain in JSON format following the instructions above.
        '''
        
        # Depending on the model, send the request to the correct service
        if self.chain_modal == "gpt":
            chain_modal = MyApi(self.api_key)
            response = chain_modal.talk(prompt)
        elif self.chain_modal == 'Deepseek':
            response = self.modal.chat(prompt)
            print(response)
        else:
            print("未识别的模型类型.")
            return None
        
        # 检查response是否为有效字符串
        if not response or len(response.strip()) == 0:
            print("未获得有效的响应.")
            return None
        
        # 使用正则表达式提取JSON部分
        json_pattern = re.compile(r'\{.*\}', re.DOTALL)
        match = json_pattern.search(response)
        
        if match:
            json_str = match.group()
            try:
                # 尝试将字符串转换为JSON对象
                reasoning_chain = json.loads(json_str)
                print(reasoning_chain)  # 打印解析结果
                return reasoning_chain
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                return None
        else:
            print("未找到有效的JSON内容。")
            return None

                

    def build_query_graph(self, data):
        """
        构建查询图：将每个步骤表示为图的节点，依赖关系表示为有向边。
        """
        for query in data:
            subquery = query['subquery']
            query_graph = {i: {'dependencies': [], 'level': None} for i in range(len(subquery))}

            # 建立依赖关系图
            for i, step in enumerate(subquery):
                value = step.get("value2")
                if isinstance(value, str) and value.startswith("IntermediateResult"):
                    try:
                        result_id = int(value.replace("IntermediateResult", ""))
                        query_graph[i]['dependencies'].append(result_id)
                    except (ValueError, IndexError) as e:
                        print(f"[Warning] 无法获取中间结果 {value}: {e}")
            
            # 执行拓扑排序，计算每个步骤的层级
            self.topological_sort(query_graph)

            # 保存查询图
            query["dependency"] = query_graph

        return data

    def topological_sort(self, query_graph):
        """
        使用拓扑排序计算查询步骤的层级。
        """
        visited = set()
        stack = []
        
        # 深度优先遍历并进行拓扑排序
        def dfs(node):
            if node not in visited:
                visited.add(node)
                for dep in query_graph[node]['dependencies']:
                    dfs(dep)
                stack.append(node)

        for node in query_graph:
            if node not in visited:
                dfs(node)

        stack.reverse() 
        for level, node in enumerate(stack):
            query_graph[node]['level'] = level
        
    def load_query_sources(self):
        Generate_emd = False

        with open(self.query_folder, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        if Generate_emd:
            embedding_model = EmbeddingModel(device='cuda:2') #"sentence-transformers/bert-base-nli-mean-tokens"
            for entry in data:
                for step_content in entry.get('subquery', []):
                
                    for key in ['condition_attribute', 'condition_value', 'target_attribute', 'target_value']:
                        text = step_content.get(key, "")
                        if text.strip():
                            embeddings = embedding_model.generate_embedding([text])[0].tolist()
                            step_content[f'{key}_emd'] = embeddings
                        else:
                            step_content[f'{key}_emd'] = []

            with open(self.query_folder, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        
        for query in data:
            for step in query['subquery']:
                for attribute in ['condition_attribute_emd', 'condition_value_emd', 'target_attribute_emd', 'target_value_emd']:
                    if step.get(attribute):
                        step[attribute] = np.array(step[attribute])

        #return self.build_query_graph(data)
        return data
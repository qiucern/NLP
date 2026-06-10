from os import path
import xml.etree.ElementTree as ET
import json
import os

def parse_semeval_xml(xml_file_path):

    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    parsed_data = [] #用于存放所有的有效句子
    
    # 极性映射表
    polarity_map = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    for sentence in root.findall('sentence'):
        text = sentence.find('text').text
        aspect_terms = [] #用于存放所有有效句子里的情绪表达
        
        # 查找所有的评价对象
        aspects_node = sentence.find('aspectTerms')
        if aspects_node is not None:
            for aspect in aspects_node.findall('aspectTerm'):
                term = aspect.get('term')
                polarity = aspect.get('polarity')
                start_idx = int(aspect.get('from'))
                end_idx = int(aspect.get('to'))
                
                if polarity == 'conflict':
                    continue
                    
                aspect_terms.append({
                    'term': term,
                    'polarity': polarity_map[polarity],
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
        
        if len(aspect_terms) > 0:
            parsed_data.append({
                'text': text,
                'aspects': aspect_terms
            })
            
    print(f"解析完成，共提取有效句子数量: {len(parsed_data)}\n")
    return parsed_data

def save_to_json(data, output_path):
    """将解析后的数据保存为 JSON 格式"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"数据已保存至: {output_path}")

# ================= 运行测试 =================
if __name__ == "__main__":
    current_file_path=os.path.abspath(__file__)
    current_dir_path=os.path.dirname(current_file_path)

    laptop_train_path = os.path.abspath(os.path.join(current_dir_path,'..','data','raw','Laptops_Train.xml'))
    restaurant_train_path = os.path.abspath(os.path.join(current_dir_path,'..','data','raw','Restaurants_Train.xml'))
    
    # 解析数据
    laptop_data = parse_semeval_xml(laptop_train_path)
    restaurant_data = parse_semeval_xml(restaurant_train_path)
    
    # 保存到 processed 文件夹
    save_to_json(laptop_data, os.path.abspath(os.path.join(current_dir_path,'..','data','processed','laptops.json')))
    save_to_json(restaurant_data, os.path.abspath(os.path.join(current_dir_path,'..','data','processed','restaurants.json')))
import os
import json

# model_path = '/raid/sjl/model/ChatGLM-v2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
# model = model.eval()
from tqdm import tqdm
import openai

# from openai import OpenAI
# client = OpenAI()

# response = client.chat.completions.create(
#   model="gpt-4-1106-preview",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Who won the world series in 2020?"},
#     {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#     {"role": "user", "content": "Where was it played?"}
#   ]
# )



def chat_with_model(templ):
    # Set up your OpenAI API credentials
    
    # Define the parameters for the conversation
    # tokenizer = "gpt-4-1106-preview"
    # history = []
    max_length = 2048
    
    # Call the chat endpoint of the OpenAI API

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        # stream=True,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": templ}
        ],
    )
    # out = ''
    # for i in response:
    #     try:
    #         out += i['choices'][0]['delta']['content']
    #     except Exception as e:
    #         # 最后一个i的delta对应的dict为空，要忽略
    #         break
    # print(out)
    
    # return out
    return response.choices[0].message.content

# Example usage
templ = "Hello, how are you?"
response = chat_with_model(templ)
print(response)

def load_json_dataset(jsonl_file_path):
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            json_data = json.loads(line)
            data.append(json_data)
    return data


idx = 0
response_list = []
data ={}
data_path = 'C:\project\socialIQa_v1.4_trn.jsonl'
file_name = 'C:\project\socialIQa_v1.4_trn_CQAF_11_27-3.jsonl' #通过扩展名指定文件存储的数据为json格式
dataset = load_json_dataset(data_path)
for example in tqdm(dataset, desc="Testing Progress"):
    idx += 1
    if idx <= 6068:
        continue
    if idx == 10000:
        break
    
    
    templ = f"""
I have a logic reasoning question that requires evaluating three possible answers based on a given scenario. 
Please provide a rationality score for each answer, ranging from 0 to 1, based on their relevance to the scenario. 
A score of 0 indicates complete irrelevance, while 1 indicates complete relevance. 
Importantly, ensure that the marked correct answer (label answer) receives the highest score. 
Avoid using extreme values (i.e., 0 or 1) unless an answer is completely irrelevant or completely relevant to the question. 
Below are the scenario, question, and label answer: {example} 
Please explain the likelihood of what will happen and provide a probability score.
Once you have derived the explanation and the score, you must output in the following format:
'''
["context:":"{example["context"]}","question:":"{example["question"]}","A":"{example["answerA"]}","Explanation_A":"$The explanation you provided.","Scores_A":"$The score you assigned.","B":"{example["answerB"]}","Explanation_B":"$The explanation you provided.","Scores_B":"$The score you assigned.","C":"{example["answerC"]}","Explanation_C":"$The explanation you provided.","Scores_C":"$The score you assigned.","correct": "{example["correct"]}"]
'''
    """
    
    response =  chat_with_model(templ)
    response = str(response)
    data={"id":idx,"response":response}
    # response_list.append(data)
    
    with open(file_name,'a',encoding='utf-8') as f:
        # for data_dict in response_list:
            json.dump(data, f,indent=4, ensure_ascii=False)
            f.write('\n')
        # json.dump(response_list,file_object, indent=4, ensure_ascii=False)

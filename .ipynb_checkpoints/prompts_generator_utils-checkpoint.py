import json 
import random
UNSAFE_CONCEPTS_PATH = "./unsafe_concepts.json"
with open(UNSAFE_CONCEPTS_PATH, "r") as f:
    unsafe_concepts = json.load(f)
# unsafe_concepts = ["nsfw", "nude", "nudity",..],it should be a list of strings
SENTENCE_TEMPLATE_PATH = "./sentence_templates.json"
with open(SENTENCE_TEMPLATE_PATH, "r") as f:
    sentence_templates = json.load(f)
#sentence_templates =["a photo of a","a rendering of a","a cropped photo of the","the photo of a","a photo of a clean","a photo of a dirty","a dark photo of the","a photo of my",...],it should be a list of strings
def generate_prompts(sentence_templates, concepts, num_prompts):
    prompts = []
    
    for _ in range(num_prompts):
        # 随机选择一个句式和一个概念
        template = random.choice(sentence_templates)
        concept = random.choice(concepts)
        
        # 创建一个新的 prompt
        prompt = f"{template} {concept}"
        prompts.append(prompt)
    
    return prompts

from transformers import AutoTokenizer, PretrainedConfig,AutoModelForCausalLM

model_path = "/root/autodl-tmp/TRCE/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
def generate_opposite_concepts(model, tokenizer, unsafe_concepts, num_concepts=5, max_length=10):
    opposite_concepts = set()
    model.eval()
    
    with torch.no_grad():
        for concept in unsafe_concepts:
            prompt = f"List {num_concepts} safe and non-sensitive concepts that are the opposite of '{concept}':\n,the output should be a json format list,for example: ['cat', 'dog', 'car', 'tree', 'house'].Don't output anything else."
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # conduct text completion
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            # 提取生成的概念
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            try:
                content=content.strip('"').strip("'")
                concepts = json.loads(content)
                if isinstance(concepts, list):
                    for c in concepts:
                        if isinstance(c, str) and c not in unsafe_concepts:
                            opposite_concepts.add(c.strip())
            except json.JSONDecodeError:
                print(f"Failed to decode JSON from model output: {content}")
    return list(opposite_concepts)

def generate_sentence_template(model, tokenizer, num_template=10, max_length=230):
    model.eval()
    sentence_templates_list = set()
    with torch.no_grad():
        prompt = f"List {num_template} different sentence templates for text-to-image generation, the output should be a json format list,for example: ['a photo of a','a rendering of a','a cropped photo of the','the photo of a','a photo of a clean','a photo of a dirty','a dark photo of the','a photo of my',...].Don't output anything else."
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        try:
            content = content.strip('"').strip("'")
            concepts = json.loads(content)
            if isinstance(concepts, list):
                for c in concepts:
                    if isinstance(c, str) and c not in unsafe_concepts:
                        sentence_templates_list.add(c.strip())
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from model output: {content}")
    return list(sentence_templates_list)
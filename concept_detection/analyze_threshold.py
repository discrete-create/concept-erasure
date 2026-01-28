import os
import json
from typing import Dict, List
from transformers import CLIPTokenizer, CLIPTextModel
import torch
import evaluate_danger


def load_txts_from_dir(txt_dir: str) -> Dict[str, List[str]]:
	"""Load all .txt files in `txt_dir` into a dict: {filename: [sentences]}.
	Empty lines are ignored and lines are stripped.
	"""
	data = {}
	for fn in os.listdir(txt_dir):
		if not fn.endswith("COCO_prompts.txt"):
			continue
		path = os.path.join(txt_dir, fn)
		with open(path, "r", encoding="utf-8") as f:
			lines = [ln.strip() for ln in f.read().splitlines()]
			lines = [ln for ln in lines if ln]
			data[fn] = lines
	return data


def evaluate_folder_texts(
	txt_dir: str,
	tokenizer,
	text_encoder,
	device: str = "cuda",
	label_dirs: str = "./unsafe_embeddings",
	batch_size: int = 256,
	output_json: str = "results.json",
):
	"""Load .txt files, compute embeddings and danger similarity, save results to JSON.

	Result JSON structure:
	{
	  "sentences": { "file.txt": [..] },
	  "embeddings": { "file.txt": [[...], ...] },
	  "danger_scores": { "file.txt": [score1, score2, ...] }
	}
	"""
	sentences_map = load_txts_from_dir(txt_dir)

	embeddings_map = {}
	danger_map = {}

	for fn, sentences in sentences_map.items():
		if not sentences:
			embeddings_map[fn] = []
			danger_map[fn] = []
			continue

		all_embs = []
		all_scores = []

		# process in batches
		for i in range(0, len(sentences), batch_size):
			batch = sentences[i : i + batch_size]
			emb_tensor = evaluate_danger.encode_batch_prompts(batch, tokenizer, text_encoder, DEVICE=device)
			# emb_tensor: [B, D] on CPU
			emb_list = emb_tensor.numpy().tolist()

			# compute per-embedding danger score (max similarity across labels)
			for emb in emb_list:
				emb_t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
				sim_dict = evaluate_danger.prompt_to_label_similarity(emb_t, label_dirs)
				if sim_dict:
					best_label = max(sim_dict, key=sim_dict.get)
					max_sim = sim_dict[best_label]
				else:
					best_label = None
					max_sim = 0.0
				all_embs.append(emb)
				all_scores.append({"score": float(max_sim), "label": best_label})

		embeddings_map[fn] = all_embs
		danger_map[fn] = all_scores

	out = { "danger_scores": danger_map}

	# save to json
	with open(output_json, "w", encoding="utf-8") as f:
		json.dump(out, f, ensure_ascii=False, indent=2)

	return out


if __name__ == "__main__":
	print("Module provides `evaluate_folder_texts(txt_dir, tokenizer, text_encoder, ...)`.\nProvide tokenizer and text_encoder (see load_model.import_models_for_cfr).")
	CLIP_MODEL_Tokennider = "/root/autodl-tmp/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/tokenizer"
	CLIP_MODEL_Text_encoder="/root/autodl-tmp/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/text_encoder"
	DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
	evaluate_folder_texts(txt_dir="/root/autodl-tmp/datasets/Unsafe Prompts&Images Dataset/prompts",tokenizer=CLIPTokenizer.from_pretrained(CLIP_MODEL_Tokennider),text_encoder=CLIPTextModel.from_pretrained(CLIP_MODEL_Text_encoder).to(DEVICE),device=DEVICE,label_dirs="unsafe_embeddings",batch_size=32,output_json="results.json")

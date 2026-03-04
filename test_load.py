import torch

if __name__ == '__main__':
    path = 'embs.pt'
    data = torch.load(path)
    print(f'loaded {path}, keys: {list(data.keys())}')
    for concept, templates in data.items():
        print(f'Concept: {concept}')
        for templ, timesteps in templates.items():
            print(f'  Template: {templ}')
            for ti, layers in timesteps.items():
                print(f'    Step index {ti}:')
                for layer_idx, entry in layers.items():
                    tensor = entry['tensor'] if isinstance(entry, dict) else entry
                    val = entry.get('value') if isinstance(entry, dict) else None
                    print(f'      layer {layer_idx} value={val} shape={tuple(tensor.shape)} dtype={tensor.dtype}')

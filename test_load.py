import torch

if __name__ == '__main__':
    path = 'embs.pt'
    data = torch.load(path)
    print(data.keys())
    #print(data['cat']['a photo of cat']['0']['0']['value'])
    print(data)
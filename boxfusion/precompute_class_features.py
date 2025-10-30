import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import open_clip
import argparse
from tools.utils import load_clip
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--clip_path", default='./models/open_clip_pytorch_model.bin', type=str, help="Path to the CLIP model")
    parser.add_argument("--class_txt", default='./data/cubicle_classes.txt', type=str)
    parser.add_argument("--class_features", default='./data/cubicle_class_features.pt', type=str)

    args = parser.parse_args()

    with torch.no_grad():
        # torch.cuda.empty_cache()

        clip_model, preprocess = load_clip(args.clip_path)
        tokenizer = open_clip.get_tokenizer("ViT-H-14")

        text_class = np.genfromtxt(args.class_txt, delimiter='\n', dtype=str) 

        text = tokenizer(text_class)

        # text_features = []
        # for label in text:
        # text_feat = clip_model.encode_text(torch.unsqueeze(label,0).cuda())
        # text_features.append(text_feat)
        text_features = clip_model.encode_text(text.cuda())
        # torch.cuda.empty_cache()

        # text_features = np.array(text_features.cpu())

        text_features = text_features/torch.linalg.vector_norm(text_features, dim=1, keepdim=True)

        torch.save(text_features, args.class_features)
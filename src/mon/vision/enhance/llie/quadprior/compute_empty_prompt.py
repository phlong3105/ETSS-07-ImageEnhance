# This is how we compute the empty embedding
# You may need to download 'openai/clip-vit-large-patch14'

import pickle

import mon
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]

model         = FrozenCLIPEmbedder().to("cuda")
embedding     = model.encode([""]).cpu()

print(embedding)
print(embedding.shape)

with open(str(current_dir / "empty_embedding.pkl"), "wb") as f:
    pickle.dump(embedding, f)

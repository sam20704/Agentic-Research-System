import os
import logging

# Environment cleanup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Python logger cleanup
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import logging as hf_logging
from sentence_transformers import SentenceTransformer

# Hugging Face logging cleanup
hf_logging.set_verbosity_error()

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(chunks):
    return model.encode(chunks)
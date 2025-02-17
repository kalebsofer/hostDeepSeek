import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Lock
from typing import List
from schemas import ChatMessage
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.lock = Lock()
        logger.info(f"Initializing ModelLoader on device: {self.device}")

    def load_model(self):
        if self.model is None:
            start_time = time.time()
            logger.info(f"Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left",
                pad_token="<|endoftext|>",
            )
            logger.info("Tokenizer loaded successfully")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                use_safetensors=True,
            )

            if torch.cuda.is_available():
                self.model.to("cuda")
                logger.info("Model moved to GPU")

            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")

    def generate_chat(
        self, messages: List[ChatMessage], max_tokens=512, temperature=0.7
    ):
        if self.model is None:
            self.load_model()

        with self.lock:
            start_time = time.time()
            logger.info("Formatting conversation...")
            formatted_conversation = self._format_conversation(messages)

            logger.info("Tokenizing input...")
            inputs = self.tokenizer(
                formatted_conversation,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            logger.info("Generating response...")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                )
            logger.info(f"Response generated in {time.time() - start_time:.2f} seconds")
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _format_conversation(self, messages: List[ChatMessage]) -> str:
        """Format conversation history into a single string"""
        formatted = []
        for msg in messages:
            if msg.role == "user":
                formatted.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted.append(f"Assistant: {msg.content}")
        return "\n".join(formatted)

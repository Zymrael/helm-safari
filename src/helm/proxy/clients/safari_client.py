"""
SafariServer handles config and model, tokenizer init and processes requests.
SafariClient preprocesses requests and sends them to SafariServer.

Note:
    The client and server below may not follow the same design patterns as others, they
    have been patched in to extend HELM evals to local models trained with the safari codebase.
"""

import torch
import sys

import json
import yaml
from copy import deepcopy
from dataclasses import asdict
from typing import List, Dict, Any

from helm.common.cache import Cache, CacheConfig
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time, truncate_sequence
from .huggingface_tokenizer import HuggingFaceTokenizers
from helm.common.hierarchical_logger import htrack_block, hlog

import os

SAFARI_PATH = os.environ.get("SAFARI_PATH", ".")
sys.path.append(SAFARI_PATH)

from src.models.sequence.long_conv_lm import ConvLMHeadModel
from src.utils.generation import generate
from transformers import AutoModelForCausalLM, GPT2Tokenizer

from src.utils import registry
from src.utils.config import instantiate

model_name_to_ckpt_path = {"safari/badger-150m": "/mnt/checkpoints/honey-badger150M-300B.ckpt"}


class SafariServer:
    def __init__(self, model_name: str):
        self.device: str = "cuda:0"

        self.model_name = model_name
        org, model_name = model_name.split("/")
        self.config = self.load_config(model_name)
        self.model, self.tokenizer = self.initialize_model_and_tokenizer(self.config)

    def load_config(self, model_name: str):
        path = os.path.join("./configs/models/", f"{model_name}.yaml")
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        return config

    def initialize_model_and_tokenizer(self, model_config: Dict):
        print(model_config)
        model = ConvLMHeadModel(**model_config["model_config"]).to(self.device)

        ckpt = torch.load(model_name_to_ckpt_path[self.model_name], map_location=self.device)
        state_dict = ckpt["state_dict"]
        state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if "model." in k}
        model.load_state_dict(state_dict)

        self.l_max = model_config["model_config"]["layer"]["l_max"]
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        postprocess_cfg = model_config.get("postprocess_cfg", {})
        if len(postprocess_cfg) > 0:
            print("Postprocessing model...")
            postprocess_fn = instantiate(registry.postprocess_methods, postprocess_cfg)
            postprocess_fn.process(model, model_config["model_config"])

        return model, tokenizer

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer

    def serve_request(self, raw_request: Dict[str, Any]):
        encoded_input = self.tokenizer(
            raw_request["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=self.l_max,
        ).to(self.device)

        raw_request = deepcopy(raw_request)
        raw_request["do_sample"] = True
        raw_request["return_dict_in_generate"] = True
        raw_request["output_scores"] = True

        top_k_per_token: int = raw_request["top_k_per_token"]
        del raw_request["top_k_per_token"]

        if len(raw_request["stop_sequences"]) > 0:
            stop_sequence_ids = self.tokenizer(raw_request["stop_sequences"])
            # Total number of stop words should be 1.
            assert len(stop_sequence_ids.input_ids) == 1
            # Total number of tokens in each stop word should be 1.
            assert len(stop_sequence_ids.input_ids[0]) == 1
            del raw_request["stop_sequences"]
            raw_request["eos_token_id"] = stop_sequence_ids.input_ids[0][0]

        # Strip out irrelevant parameters
        relevant_raw_request = {
            key: raw_request[key]
            for key in raw_request
            if key not in ["engine", "prompt", "echo_prompt", "stop_sequences", "do_sample"]
        }

        # Use Safari's `generate` method.
        encoded_input = encoded_input.input_ids
        output = generate(self.model, encoded_input, self.l_max, **relevant_raw_request)

        if relevant_raw_request["output_scores"]:
            sequences, scores = output
        else:
            sequences = output
            scores = None

        # Compute logprobs for each completed sequence.
        all_logprobs_of_chosen_tokens = []
        all_top_logprobs_dicts = []
        for completion_id in range(raw_request["num_return_sequences"]):
            logprobs_of_chosen_tokens = []
            top_logprobs_dicts = []
            for i in range(len(sequences[completion_id]) - len(encoded_input[0])):
                logprobs = torch.nn.functional.log_softmax(scores[completion_id, i], dim=0)

                # Get top tokens in terms of log probability.
                topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                top_logprobs_dicts.append(
                    {
                        self.tokenizer.convert_ids_to_tokens(k.item()): v.item()
                        for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                    }
                )

                # Get log probability of chosen token.
                j = i + len(encoded_input[0])
                logprobs_of_chosen_tokens.append(logprobs[sequences[completion_id][j]].item())
            all_logprobs_of_chosen_tokens.append(logprobs_of_chosen_tokens)
            all_top_logprobs_dicts.append(top_logprobs_dicts)

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input[0]) :] for sequence in sequences]

        # TODO: Get rid of the extra tokenization?
        all_tokens = [self.tokenizer.convert_ids_to_tokens(sequence) for sequence in sequences]
        all_decoded_text = self.tokenizer.batch_decode(sequences)

        completions = []
        for decoded_text, tokens, logprobs_of_chosen_tokens, top_logprobs_dicts in zip(
            all_decoded_text, all_tokens, all_logprobs_of_chosen_tokens, all_top_logprobs_dicts
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": logprobs_of_chosen_tokens,
                    "top_logprobs_dicts": top_logprobs_dicts,
                }
            )

        # pmp = raw_request["prompt"]
        # print(f"Prompt: {pmp}, Completion: {all_decoded_text}")
        return {"completions": completions, "input_length": len(encoded_input[0])}


class SafariClient(Client):
    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)
        self.server = None
        self.model_server_instance = None

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model

        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        # Only a single stop sequence is supported as we can only pass in a single value for `eos_token_id`
        if len(request.stop_sequences) > 1:
            raise ValueError("More than one stop sequence is not supported.")

        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences,
        }

        # Get cached model server instance if possible (to save on model and tokenizer
        # loading times).
        if self.model_server_instance is None:
            model_server_instance = SafariServer(request.model)
            self.model_server_instance = model_server_instance

        try:

            def do_it():
                return self.model_server_instance.serve_request(raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            # response = wrap_request_time(do_it)()
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:  # Do something if error is encountered.
            error: str = f"Safari error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            if request.echo_prompt:
                # Add prompt to list of generated tokens.
                generated_tokens = raw_completion["tokens"][response["input_length"] :]
                for token_text in raw_completion["tokens"][: response["input_length"]]:
                    tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))
            else:
                generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob, top_logprobs_dict in zip(
                generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
            ):
                tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
                sequence_logprob += logprob

            completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)

            completion = truncate_sequence(completion, request)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        return super().tokenize(request)

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        return super().decode(request)

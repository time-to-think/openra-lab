"""Trainable LLM policy: one local base model shared by several LoRA adapters, one adapter per player.

Each adapter is updated with a policy-gradient step (REINFORCE with batch-normalized advantages) plus a KL
penalty that keeps it close to the frozen base model, which is the base model with its adapters disabled.
Heavy dependencies are imported lazily so the arena and its offline tests run without them.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl.buffer import Transition

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class PolicyConfig:
    base_model: str = "Qwen/Qwen3-1.7B"
    checkpoint_dir: Path = Path("checkpoints")
    lora_rank: int = 16
    learning_rate: float = 2e-5
    kl_beta: float = 0.05
    temperature: float = 1.0
    max_new_tokens: int = 320
    max_sequence: int = 4096


class LoraPolicy:
    def __init__(self, cfg: PolicyConfig, adapters: list[str]) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg, self.torch, self.lock = cfg, torch, threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("[policy] CUDA is unavailable; training on the CPU will be very slow.", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=dtype).to(self.device)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lora = LoraConfig(r=cfg.lora_rank, lora_alpha=2 * cfg.lora_rank, lora_dropout=0.0,
                          target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        self.model: Any = get_peft_model(base, lora, adapter_name=adapters[0])
        for name in adapters[1:]:
            self.model.add_adapter(name, lora)
        self.optimizers: dict[str, Any] = {}
        self.updates: dict[str, int] = {}
        for name in adapters:
            self.optimizers[name] = torch.optim.AdamW(self._params(name), lr=cfg.learning_rate, weight_decay=0.0)
            self.updates[name] = self._load(name)

    def generate(self, adapter: str, system: str, prompt: str) -> tuple[str, list[int]]:
        torch = self.torch
        with self.lock:
            self._activate(adapter, train=False)
            ids = torch.tensor([self._encode(system, prompt)], device=self.device)
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                    temperature=self.cfg.temperature, top_p=1.0, max_new_tokens=self.cfg.max_new_tokens,
                    use_cache=True, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            tokens = out[0, ids.shape[1]:].tolist()
            return self.tokenizer.decode(tokens, skip_special_tokens=True), tokens

    def update(self, adapter: str, batch: list[Transition]) -> dict[str, float]:
        """One policy-gradient step over `batch`; returns training metrics."""
        torch = self.torch
        with self.lock:
            self._activate(adapter, train=True)
            optimizer = self.optimizers[adapter]
            optimizer.zero_grad(set_to_none=True)
            loss_sum = kl_sum = 0.0
            samples = [t for t in batch if t.response_ids]
            for t in samples:
                ids = self._sequence(t)
                n = len(t.response_ids)
                logp = self._token_logp(ids, n)
                with torch.no_grad(), self.model.disable_adapter():
                    ref = self._token_logp(ids, n)
                diff = ref - logp
                kl = (torch.exp(diff) - diff - 1).mean()
                loss = (-t.advantage * logp.mean() + self.cfg.kl_beta * kl) / len(samples)
                loss.backward()
                loss_sum += loss.item()
                kl_sum += kl.item() / len(samples)
            grad_norm = torch.nn.utils.clip_grad_norm_(self._params(adapter), 1.0).item()
            optimizer.step()
            self.updates[adapter] += 1
            self._save(adapter)
            return {"update": self.updates[adapter], "loss": round(loss_sum, 5), "kl": round(kl_sum, 5),
                    "grad_norm": round(grad_norm, 4), "samples": len(samples)}

    def _token_logp(self, ids: Any, n: int) -> Any:
        """Log-probabilities of the last `n` tokens; only those logits are materialized."""
        logits = self.model(input_ids=ids, logits_to_keep=n + 1).logits[:, :-1].float()
        return self.torch.log_softmax(logits, -1).gather(-1, ids[:, -n:].unsqueeze(-1)).squeeze(-1)

    def _sequence(self, t: Transition) -> Any:
        prompt = self._encode(t.system, t.prompt)
        keep = max(1, self.cfg.max_sequence - len(t.response_ids))
        return self.torch.tensor([prompt[-keep:] + list(t.response_ids)], device=self.device)

    def _encode(self, system: str, prompt: str) -> list[int]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        # enable_thinking is read by Qwen3 templates and ignored by others; actions must be short and direct.
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _activate(self, adapter: str, train: bool) -> None:
        self.model.set_adapter(adapter)
        for name, param in self.model.named_parameters():
            param.requires_grad = train and f".{adapter}." in name
        self.model.train(train)

    def _params(self, adapter: str) -> list[Any]:
        return [p for name, p in self.model.named_parameters() if f".{adapter}." in name]

    def _path(self, adapter: str) -> Path:
        return self.cfg.checkpoint_dir / adapter

    def _save(self, adapter: str) -> None:
        from peft import get_peft_model_state_dict

        path = self._path(adapter)
        path.mkdir(parents=True, exist_ok=True)
        weights = get_peft_model_state_dict(self.model, adapter_name=adapter)
        self.torch.save({"weights": weights, "optimizer": self.optimizers[adapter].state_dict()}, path / "adapter.pt")
        (path / "state.json").write_text(json.dumps({
            "base_model": self.cfg.base_model, "lora_rank": self.cfg.lora_rank, "updates": self.updates[adapter],
        }, indent=2), encoding="utf-8")

    def _load(self, adapter: str) -> int:
        """Resume an adapter trained in earlier sessions; returns its number of completed updates."""
        from peft import set_peft_model_state_dict

        path = self._path(adapter)
        if not (path / "adapter.pt").exists():
            return 0
        meta = json.loads((path / "state.json").read_text(encoding="utf-8"))
        if meta.get("base_model") != self.cfg.base_model or meta.get("lora_rank") != self.cfg.lora_rank:
            raise ValueError(f"{path} was trained for {meta.get('base_model')} rank {meta.get('lora_rank')}; "
                             "choose another --checkpoint-dir or the matching --base-model/--lora-rank.")
        saved = self.torch.load(path / "adapter.pt", map_location=self.device)
        set_peft_model_state_dict(self.model, saved["weights"], adapter_name=adapter)
        self.optimizers[adapter].load_state_dict(saved["optimizer"])
        print(f"[policy] resumed adapter {adapter} after {meta.get('updates', 0)} updates", flush=True)
        return int(meta.get("updates", 0))

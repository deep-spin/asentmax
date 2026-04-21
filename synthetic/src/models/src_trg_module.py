import torch
from torch import nn
from typing import Tuple
from omegaconf import DictConfig
from pathlib import Path

from .base_module import (
    BaseLitModule
)


class SrcTrgLitModule(BaseLitModule):
    """LightningModule for source-to-target sequence generation tasks.

    Extends BaseLitModule with autoregressive generation, EOS-based truncation,
    and optional attention weight saving. The loss can be computed over the
    target tokens only or over the entire input sequence.
    """

    remove_from_checkpoint = ["hyper_parameters.metrics"]

    def __init__(self,
                 tokenizer: nn.Module,
                 criterion: nn.Module,
                 metrics: DictConfig,
                 metrics_config: DictConfig,
                 optimizer: DictConfig,
                 scheduler: DictConfig,
                 net: nn.Module,
                 checkpoint: nn.Module = None,
                 checkpoint_type: str = None,
                 generation: DictConfig = None,
                 loss_type: str = 'all',
                 generation_multiplier: float = 3,
                 generate_max_seq=False,
                 save_attn_to: str = None,
                 **kwargs
                 ):
        """Initialize SrcTrgLitModule.

        :param tokenizer: Tokenizer used for encoding/decoding text.
        :param criterion: Loss function (e.g. CrossEntropyLoss).
        :param metrics: Dict of metric modules.
        :param metrics_config: Config specifying which metrics to use per stage.
        :param optimizer: Partially instantiated optimizer.
        :param scheduler: Scheduler config with ``instance`` and ``params``.
        :param net: Partially instantiated model (receives ``vocab_size`` at init).
        :param checkpoint: Optional checkpoint module for weight loading.
        :param checkpoint_type: Type of checkpoint format.
        :param generation: Generation config (max_length, num_beams, etc.).
        :param loss_type: ``"target"`` to compute loss on target tokens only,
            ``"all"`` to compute loss on the entire sequence.
        :param generation_multiplier: Fallback max generation length as a
            multiple of source length (used when ``max_length`` is not set).
        :param generate_max_seq: If True, generate up to ``max_length`` tokens
            ignoring EOS (useful for fixed-length output tasks).
        :param save_attn_to: Directory path to save attention weights. If None,
            attention weights are not saved.
        """
        self.tokenizer = tokenizer

        if loss_type not in {"target", "all"}:
            raise ValueError(f"loss_type `{loss_type}` not supported")

        # Net is partially instantiated because we need to pass vocab_size
        net = net(vocab_size=len(tokenizer))

        if generate_max_seq:
            if generation is None or 'max_length' not in generation:
                raise ValueError("`generate_max_seq=True` requires `generation.max_length` to be set")

        super().__init__(
            net=net,
            criterion=criterion,
            metrics=metrics,
            metrics_config=metrics_config,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint=checkpoint,
            checkpoint_type=checkpoint_type,
            generation=generation,
            loss_type=loss_type,
            generation_multiplier=generation_multiplier,
            generate_max_seq=generate_max_seq,
            save_attn_to=save_attn_to,
            **kwargs
        )

    def model_step(
            self,
            stage_name: str,
            batch: dict[str, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = None,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss and optionally generate predictions for a single batch.

        During training, only the forward pass and loss computation are performed.
        During validation/test, autoregressive generation is also run and
        predictions are post-processed (EOS truncation + decoding).

        :param stage_name: Stage name (``"train"``, ``"val"``, or ``"test"``).
        :param batch: Batch dict with keys ``input_ids``, ``input_mask``,
            ``target_mask``, ``targets``, ``source_ids``, ``source_mask``,
            and ``target_str``.
        :param batch_idx: Index of the current batch.
        :param dataloader_idx: Index of the current dataloader.
        :return: Tuple of (loss, predictions, targets). Predictions are None
            during training.
        """
        if self.hparams.loss_type == 'target':
            loss_mask = batch['target_mask']
        elif self.hparams.loss_type == 'all':
            loss_mask = batch['input_mask']
        else:
            raise ValueError(f"Unexpected loss_type `{self.hparams.loss_type}`")

        targets = batch['targets']

        self.log(
            'seq_len',
            batch['input_ids'].shape[1],
            batch_size=batch['input_ids'].shape[0],
            sync_dist=self.sync_dist
        )

        out = self.net(batch['input_ids'], batch['input_mask'], **kwargs)
        loss = self.criterion(out.logits[loss_mask], targets[loss_mask])

        preds = None
        if stage_name != 'train':
            preds = self.batch_generate(batch['source_ids'], batch['source_mask'])
            # Strip source prefix from generated output
            preds = preds[:, batch['source_ids'].shape[1]:]
            preds = self.post_process_preds(preds)
            batch['preds'] = preds

        if self.hparams.get("tokenizer_decode", True):
            result = loss, preds, batch['target_str']
        else:
            result = loss, preds, batch['targets']

        if (self.hparams.save_attn_to is not None
                and hasattr(out, "attentions")
                and out.attentions is not None):
            self.save_attention_weights(out.attentions, batch['input_ids'], dataloader_idx, batch_idx)

        return result

    def batch_generate(self, source_ids, source_mask):
        """Run autoregressive generation on a batch of source sequences.

        Resolves ``max_length`` from config or falls back to
        ``source_length * generation_multiplier``. Optionally disables
        EOS-based stopping when ``generate_max_seq`` is True.

        :param source_ids: Source token IDs, shape ``(batch_size, source_len)``.
        :param source_mask: Attention mask for source tokens.
        :return: Generated token IDs including the source prefix.
        """
        gen_params = dict(self.hparams.generation)

        if 'max_length' in self.hparams.generation and self.hparams.generation['max_length'] is not None:
            gen_params['max_length'] += source_ids.shape[1]
        else:
            gen_params['max_length'] = int(source_ids.shape[1] * self.hparams.generation_multiplier)

        if 'add_pad_token_id' in gen_params and gen_params['add_pad_token_id']:
            gen_params['pad_token_id'] = self.tokenizer.pad_token_id
            del gen_params['add_pad_token_id']

        if self.hparams.generate_max_seq:
            # Use an impossible token ID so generation never stops at EOS
            eos_token_id = -1000
        else:
            eos_token_id = self.tokenizer.eos_token_id

        output_ids = self.net.generate(
            source_ids,
            input_mask=source_mask,
            eos_token_id=eos_token_id,
            **gen_params
        )

        return output_ids

    def post_process_preds(self, preds):
        """Truncate predictions at the first EOS token and decode to strings.

        :param preds: Generated token IDs, shape ``(batch_size, seq_len)``.
        :return: List of decoded prediction strings.
        """
        eos_mask = (preds == self.tokenizer.eos_token_id).cumsum(dim=1) > 0
        eos_pos = (~eos_mask).cumsum(dim=1)[:, -1]
        preds = [preds[i, :pos].cpu().tolist() for i, pos in enumerate(eos_pos)]
        preds = [self.tokenizer.decode(p) for p in preds]

        return preds

    def save_attention_weights(self, attentions, input_ids, dataloader_idx, batch_idx):
        """Save attention weights and input IDs to disk as a .pt file.

        Files are saved to ``self.hparams.save_attn_to`` with naming convention
        ``loader_{idx}_batch_{idx}.pt``.

        :param attentions: Attention weight tensors from the model.
        :param input_ids: Input token IDs corresponding to the attention weights.
        :param dataloader_idx: Index of the current dataloader.
        :param batch_idx: Index of the current batch.
        """
        out_dir = Path(self.hparams.save_attn_to)
        out_dir.mkdir(exist_ok=True)

        loader_idx = dataloader_idx if dataloader_idx is not None else 0
        file_path = out_dir / f"loader_{loader_idx}_batch_{batch_idx}.pt"

        torch.save({"input_ids": input_ids, "attentions": attentions}, file_path)


if __name__ == "__main__":
    pass

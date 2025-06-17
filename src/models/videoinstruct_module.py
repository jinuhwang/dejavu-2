from functools import partial
from typing import Any, Dict, List, Tuple

from box import Box
import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from ..utils.paths import get_path_manager
from ..utils.dataset import save_embedding


class VideoinstructLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        blobnet_lr: float,
        restoration_lr: float,
        decision_lr: float,
        rloss_pattern: List[bool],
        sloss_pattern: List[bool],
        compile: bool,
        gating_hard: bool,
        similarity_lr: float = 0.,
        batch_size: int = 1,
        dataset: str = 'nextqa',
        fps: int = 2,
        test_split: str = 'val',
        base_model_name: str = 'openai/clip-vit-large-patch14',
        dry_run: bool = True,
        mode: str = 'original',
        eventful_topk: int = 1,
        cmc_threshold: float = 170,
        reuse_model_name: str = None,
        debug: bool = False,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.reuse_loss = loss

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_hidden_error = MeanMetric()
        self.train_hh_error = MeanMetric() # Second to last layer hidden error
        self.train_cls_error = MeanMetric()
        self.train_reuse_rate = MeanMetric()
        # validation metrics
        self.val_loss = MeanMetric()
        self.val_hidden_error = MeanMetric()
        self.val_hh_error = MeanMetric() # Second to last layer hidden error
        self.val_cls_error = MeanMetric()
        self.val_reuse_rate = MeanMetric()
        self.val_loss_best = MinMetric()
        # test metrics
        self.test_sim = MeanMetric()
        self.test_reuse_rate = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_hidden_error.reset()
        self.val_hh_error.reset()
        self.val_cls_error.reset()
        self.val_reuse_rate.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        (
            frame_idxs,
            pixel_values,
            compressed,
            ref_type,
            ref_mask,
        ) = batch
        output, reuse_maps, hidden_states = self.net(
            pixel_values,
            compressed=compressed,
            ref_type=ref_type,
            ref_mask=ref_mask,
            output_hidden_states=True,
            hard=self.hparams.gating_hard,
        )

        if self.hparams.sloss_pattern is not None:
            pixel_values = pixel_values[:, self.hparams.sloss_pattern]
            hidden_states = [h[:, self.hparams.sloss_pattern] for h in hidden_states]
            output = output[:, self.hparams.sloss_pattern]

        if self.hparams.rloss_pattern is not None:
            reuse_maps = reuse_maps[:, self.hparams.rloss_pattern]

        with torch.no_grad():
            B, F, C, H, W = pixel_values.shape
            pixel_values = pixel_values.view(B * F, C, H, W)
            original_output = self.net.orig_model(
                pixel_values,
                output_hidden_states=True,
            )
            original_hidden_states = [o.view(B, F, *o.shape[1:]) for o in original_output.hidden_states]

            if hasattr(original_output, 'image_embeds'):
                original_output = original_output.image_embeds
            else:
                original_output = original_output.pooler_output

            original_output = original_output.view(B, F, -1)

        return self.reuse_loss(
            hidden_states,
            output,
            original_hidden_states,
            original_output,
            reuse_maps,
        )

    def test_model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            is_new_videos,
            pixel_values,
            compressed,
        ) = batch

        assert is_new_videos.any() == is_new_videos.all(), 'batch size must be 1 for now'
        is_new_video = is_new_videos[0].item()

        if self.hparams.mode == 'original':
            inference_outputs = self.net(
                pixel_values=pixel_values,
                compressed=compressed,
                disable_reuse=is_new_video,
                output_hidden_states=True,
                output_maps=True,
                **self.kwargs
            )
        elif self.hparams.mode in ['eventful', 'cmc', 'reuse-sequential', 'reuse-train']:
            outputs, maps, hidden_states, cached_states_next = self.net.forward_eval(
                pixel_values=pixel_values,
                compressed=compressed,
                disable_reuse=is_new_video,
                output_hidden_states=True,
                output_maps=True,
                cached_states_prev=None if is_new_video else self.cached_states,
                ref_mask=self.ref_mask,
                ref_type=self.ref_type,
                next_states_idx=0 if self.hparams.mode == 'reuse-train' else None,
            )

            self.cached_states = cached_states_next

            if is_new_video:
                maps.fill_(0)

            inference_outputs = Box({
                'image_embeds': outputs,
                'maps': maps,
                'hidden_states': hidden_states,
            })
            if self.hparams.debug:
                original_output = self.net.orig_model(
                    pixel_values[0],
                    output_hidden_states=True,
                )
        else:
            raise NotImplementedError(f'Unknown mode: {self.hparams.mode}')

        return inference_outputs

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, hidden_error, hh_error, cls_error, reuse_rate, reuse_rate_per_frame = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_hidden_error(hidden_error)
        self.train_hh_error(hh_error)
        self.train_cls_error(cls_error)
        self.train_reuse_rate(reuse_rate)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/hidden_error", self.train_hidden_error, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/hh_error", self.train_hh_error, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/cls_error", self.train_cls_error, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/reuse_rate", self.train_reuse_rate, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        # Log the learning rate
        blobnet_lr = self.optimizers().param_groups[0]['lr']
        restoration_lr = self.optimizers().param_groups[1]['lr']
        decision_lr = self.optimizers().param_groups[2]['lr']
        similarity_lr = self.optimizers().param_groups[3]['lr'] 
        self.log("lr/blobnet", blobnet_lr, on_step=False, on_epoch=True, prog_bar=False)
        self.log("lr/restoration", restoration_lr, on_step=False, on_epoch=True, prog_bar=False)
        self.log("lr/decision", decision_lr, on_step=False, on_epoch=True, prog_bar=False)
        self.log("lr/similarity", similarity_lr, on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, hidden_error, hh_error, cls_error, reuse_rate, reuse_rate_per_frame = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_hidden_error(hidden_error)
        self.val_hh_error(hh_error)
        self.val_cls_error(cls_error)
        self.val_reuse_rate(reuse_rate)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/hidden_error", self.val_hidden_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/hh_error", self.val_hh_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/cls_error", self.val_cls_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/reuse_rate", self.val_reuse_rate, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        (
            video_ids,
            is_new_videos,
            frame_idxs,
            pixel_values,
            original_outputs,
            compressed,
        ) = batch

        batch = (is_new_videos, pixel_values, compressed)

        inference_outputs = self.test_model_step(batch)
        image_embeds = inference_outputs.image_embeds.squeeze()
        original_outputs = original_outputs.squeeze(0)
        maps = inference_outputs.maps

        sim = torch.cosine_similarity(image_embeds, original_outputs, dim=-1)
        
        reuse_rates = []
        # Iterate over layers
        for m in maps:
            if m is None:
                continue
            m = m.float()
            reuse_rates.append(m.mean(dim=(-1, -2)))
        reuse_rates = torch.mean(torch.stack(reuse_rates), dim=0)

        self.log("test/sim", sim.mean(), on_step=True, on_epoch=False, prog_bar=True)
        self.log("test/reuse_rate", reuse_rates.mean(), on_step=True, on_epoch=False, prog_bar=True)

        # Save embeddings
        video_id = video_ids[0]

        unique_idx = []
        for idx, frame_idx in enumerate(frame_idxs.flatten()):
            if frame_idx in unique_idx:
                continue
            unique_idx.append(frame_idx)

            embedding = image_embeds[idx]
            embedding_path = self.get_feature_path(
                video_id=video_id,
                frame_num=frame_idx,
            )
            if self.hparams.dry_run:
                print(f"DRY RUN: Saving embedding to {embedding_path}")
            else:
                save_embedding(embedding, embedding_path)
            
            self.test_sim.update(sim[idx])
            self.test_reuse_rate.update(reuse_rates[idx])

        self.log("test/sim_mean", self.test_sim.compute(), on_step=True, on_epoch=False, prog_bar=True)
        self.log("test/reuse_rate_mean", self.test_reuse_rate.compute(), on_step=True, on_epoch=False, prog_bar=True)


    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

        if stage in ['predict', 'test']:
            batch_size = self.hparams.batch_size

            get_feature_path_kwargs = {
                'feature_type': 'o',
                'dataset': self.hparams.dataset,
                'fps': self.hparams.fps,
                'split': self.hparams.test_split,
                'base_model_name': self.hparams.base_model_name,
                'use_v2': True,
            }

            path_manager = get_path_manager()
            if self.hparams.mode in ['original', 'reuse-sequential', 'reuse-train']:
                self.get_feature_path = partial(
                    path_manager.get_reuse_path,
                    reuse_model_name=self.hparams.reuse_model_name,
                    **get_feature_path_kwargs,
                )
            elif self.hparams.mode == 'eventful':
                self.get_feature_path = partial(
                    path_manager.get_eventful_path,
                    topk=self.hparams.eventful_topk,
                    **get_feature_path_kwargs,
                )
            elif self.hparams.mode == 'cmc':
                self.get_feature_path = partial(
                    path_manager.get_cmc_path,
                    threshold=self.hparams.cmc_threshold,
                    **get_feature_path_kwargs,
                )
            else:
                raise NotImplementedError(f'Unknown mode: {self.hparams.mode}')

            if self.hparams.mode == 'original':
                config = self.net.model.config
                num_hidden_layers = config.num_hidden_layers
                hidden_size = config.hidden_size
                N = (config.image_size // config.patch_size) ** 2 + 1

                self.kwargs = {
                    'reference_caches': torch.zeros(
                        (num_hidden_layers, batch_size, N, hidden_size),
                        device='cuda',
                        dtype=torch.float32
                    ),
                    'hqkv_caches': torch.zeros(
                        (num_hidden_layers, 4, batch_size, N, hidden_size),
                        device='cuda',
                        dtype=torch.float32
                    ),
                    'reference_type': torch.nn.functional.one_hot(
                        torch.tensor([0, 1, 2, 2], device='cuda').unsqueeze(1).expand(-1, batch_size),
                    )
                }
            elif self.hparams.mode in ['eventful', 'cmc']:
                self.ref_mask = torch.tensor(
                    [
                        [ True, False, False, False],
                        [False,  True, False, False],
                        [False, False,  True, False],
                        [False, False, False,  True],
                    ],
                    device='cuda',
                )
                self.ref_mask = self.ref_mask.unsqueeze(0).expand(batch_size, -1, -1)
                self.ref_type = None
                self.cached_states = None
            elif self.hparams.mode == 'reuse-sequential':
                self.ref_mask = torch.tensor(
                    [
                        [ True, False, False, False],
                        [ True,  True, False, False],
                        [False,  True,  True, False],
                        [False, False,  True,  True],
                    ],
                    device='cuda',
                )
                self.ref_mask = self.ref_mask.unsqueeze(0).expand(batch_size, -1, -1)
                self.ref_type = torch.nn.functional.one_hot(
                    torch.tensor([1, 1, 1, 1], device='cuda').unsqueeze(0).expand(batch_size, -1),
                    num_classes=3,
                )
                self.cached_states = None
            elif self.hparams.mode == 'reuse-train':
                self.ref_mask = torch.tensor(
                    [
                        [ True, False, False, False], # 0 => 0, 4
                        [ True,  True, False, False], # 0, 4 => 0, 4, 2
                        [ True, False,  True, False], # 0, 2 => 0, 4, 2, 1
                        [False,  True,  True, False], # 4, 2 => 0, 4, 2, 1, 3
                    ],
                    device='cuda',
                )
                self.ref_mask = self.ref_mask.unsqueeze(0).expand(batch_size, -1, -1)
                self.ref_type = torch.nn.functional.one_hot(
                    torch.tensor([0, 1, 2, 2], device='cuda').unsqueeze(0).expand(batch_size, -1),
                    num_classes=3,
                )
                self.cached_states = None

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        blobnet_params = []
        restoration_params = []
        decision_params = []
        similarity_params = []
        for n, p in self.trainer.model.named_parameters():
            if 'blobnet' in n:
                blobnet_params.append(p)
            elif 'restoration' in n:
                restoration_params.append(p)
            elif 'decision' in n:
                decision_params.append(p)
            elif 'similarity' in n:
                similarity_params.append(p)
            else:
                p.requires_grad = False

        optimizer = self.hparams.optimizer(
            [
                {'params': blobnet_params, 'lr': self.hparams.blobnet_lr},
                {'params': restoration_params, 'lr': self.hparams.restoration_lr},
                {'params': decision_params, 'lr': self.hparams.decision_lr},
                {'params': similarity_params, 'lr': self.hparams.similarity_lr},
            ],
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


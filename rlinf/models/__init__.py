# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional

from omegaconf import DictConfig

from rlinf.config import EMBODIED_MODEL, SupportedModel, torch_dtype_from_precision
from rlinf.scheduler import Worker

ModelBuilder = Callable[[DictConfig, Optional[object]], object]
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(
    model_type: str,
    model_builder: ModelBuilder,
    category: str = "embodied",
    force: bool = False,
):
    """Register a model builder for cfg.model_type."""
    if not model_type:
        raise ValueError("model_type must be a non-empty string.")
    if not callable(model_builder):
        raise TypeError("model_builder must be callable.")
    if not force and model_type in _MODEL_REGISTRY:
        raise ValueError(
            f"Model type `{model_type}` is already registered. "
            "Set force=True to override it."
        )
    _MODEL_REGISTRY[model_type] = model_builder
    SupportedModel.register(model_type, force=force)
    if category == "embodied":
        EMBODIED_MODEL.add(SupportedModel(model_type))


def _register_builtin_models():
    def _build_openvla(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openvla import get_model

        return get_model(cfg, torch_dtype)

    def _build_openvla_oft(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openvla_oft import get_model

        return get_model(cfg, torch_dtype)

    def _build_openpi(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openpi import get_model

        return get_model(cfg, torch_dtype)

    def _build_dexbotic_pi(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.dexbotic_pi import get_model

        return get_model(cfg, torch_dtype)

    def _build_dexbotic_dm0(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.dexbotic_dm0 import get_model

        return get_model(cfg, torch_dtype)

    def _build_mlp_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.mlp_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_rlt_mlp_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.mlp_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_gr00t(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.gr00t import get_model

        return get_model(cfg, torch_dtype)

    def _build_cnn_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.cnn_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_flow_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.flow_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_lingbotvla(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.lingbotvla import get_model

        return get_model(cfg, torch_dtype)

    def _build_abot_m0(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.abot_m0 import get_model

        return get_model(cfg, torch_dtype)

    def _build_starvla(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.starvla import get_model

        return get_model(cfg, torch_dtype)

    def _build_dreamzero(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.dreamzero import get_model

        return get_model(cfg, torch_dtype)

    def _build_gr00t_n1d6(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.gr00t import get_model

        return get_model(cfg, torch_dtype)

    def _build_gr00t_n1d7(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.gr00t import get_model

        return get_model(cfg, torch_dtype)

    def _build_openpi_cfg(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openpi_cfg import get_model

        return get_model(cfg, torch_dtype)

    def _build_recap_value_model(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.value_model.recap import get_model

        return get_model(cfg, torch_dtype)

    def _build_steam_value_model(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.value_model.steam import get_model

        return get_model(cfg, torch_dtype)

    register_model(
        SupportedModel.OPENVLA.value,
        _build_openvla,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.OPENVLA_OFT.value,
        _build_openvla_oft,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.OPENPI.value,
        _build_openpi,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.DEXBOTIC_PI.value,
        _build_dexbotic_pi,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.DEXBOTIC_DM0.value,
        _build_dexbotic_dm0,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.MLP_POLICY.value,
        _build_mlp_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.RLT_MLP_POLICY.value,
        _build_rlt_mlp_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.GR00T.value,
        _build_gr00t,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.CNN_POLICY.value,
        _build_cnn_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.FLOW_POLICY.value,
        _build_flow_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.LINGBOTVLA.value,
        _build_lingbotvla,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.ABOT_M0.value,
        _build_abot_m0,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.STARVLA.value,
        _build_starvla,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.DREAMZERO.value,
        _build_dreamzero,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.CFG_MODEL.value,
        _build_openpi_cfg,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.RECAP_VALUE_MODEL.value,
        _build_recap_value_model,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.STEAM_VALUE_MODEL.value,
        _build_steam_value_model,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.GR00T_N1D6.value,
        _build_gr00t_n1d6,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.GR00T_N1D7.value,
        _build_gr00t_n1d7,
        category="embodied",
        force=True,
    )


_register_builtin_models()


def get_model(cfg: DictConfig):
    model_type = str(cfg.model_type)
    model_builder = _MODEL_REGISTRY.get(model_type)
    if model_builder is None:
        return None

    torch_dtype = torch_dtype_from_precision(cfg.precision)
    model = model_builder(cfg, torch_dtype)

    if (
        Worker.torch_platform is not None
        and Worker.torch_platform.is_available()
        and cfg.get("load_to_device", True)
    ):
        model = model.to(Worker.torch_device_type)

    if cfg.is_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        if not hasattr(cfg, "lora_path") or cfg.lora_path is None:
            # A regex confines the adapters to a subset of modules. peft treats
            # a string target_modules as a regex matched with re.fullmatch.
            #
            # Why this matters for memory and speed: autograd only has to keep
            # activations for layers that lie between the loss and the deepest
            # trainable parameter. Adapting the TOP k transformer blocks means
            # everything below them -- and the vision tower feeding them --
            # runs without building a graph. Adapting anything in the vision
            # tower instead forces the whole language stack above it to be
            # retained, since the gradient has to travel back down through it,
            # so a "vision only" LoRA is the most expensive option, not the
            # cheapest.
            target_regex = cfg.get("lora_target_regex", None)
            # lora_alpha sets the adapter's output scale: the forward adds
            # (lora_alpha / r) * B @ A to the frozen weight. Default None keeps
            # alpha == r, i.e. scale 1.0, which is what every existing config
            # has been getting; the usual convention is alpha = 2r.
            lora_alpha = cfg.get("lora_alpha", None) or cfg.lora_rank
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                target_modules=target_regex
                if target_regex
                else [
                    "proj",
                    "qkv",
                    "fc1",
                    "fc2",  # vision
                    "q",
                    "kv",
                    "fc3",
                    "out_proj",  # project
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",  # llm
                ],
                init_lora_weights="gaussian",
            )
            if SupportedModel(model_type) in (
                SupportedModel.OPENPI,
                SupportedModel.CFG_MODEL,
            ):
                module_to_lora = model.paligemma_with_expert.paligemma
                module_to_lora = get_peft_model(module_to_lora, lora_config)
                tag_vlm_subtree(model, False)
                tag_vlm_subtree(module_to_lora, True)
                model.paligemma_with_expert.paligemma = module_to_lora
            elif SupportedModel(model_type) in (
                SupportedModel.GR00T,
                SupportedModel.GR00T_N1D6,
                SupportedModel.GR00T_N1D7,
            ):
                # Adapt the Eagle VLM only. get_peft_model on the whole model
                # freezes every module it does not match, and none of
                # target_modules matches the diffusers-named DiT, the
                # state/action encoders, vl_self_attention or the value head --
                # so the action head would stop training altogether and the
                # actor would never move. Wrap the backbone alone and leave the
                # head exactly as tune_projector/tune_diffusion_model left it.
                module_to_lora = get_peft_model(model.backbone, lora_config)
                tag_vlm_subtree(model, False)
                tag_vlm_subtree(module_to_lora, True)
                model.backbone = module_to_lora
            else:
                model = get_peft_model(model, lora_config)
        else:
            model = PeftModel.from_pretrained(model, cfg.lora_path, is_trainable=True)

        if hasattr(model, "value_head"):
            for param in model.value_head.parameters():
                param.requires_grad = True

        # Log what the adapters actually attached to. A silent no-op here (0
        # trainable adapter params, or a target_modules list that matched
        # nothing) looks exactly like a successful run until the numbers come
        # back identical to the frozen arm, so state it once at build time.
        from rlinf.utils.logging import get_logger

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora = sum(
            p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n
        )
        total = sum(p.numel() for p in model.parameters())
        get_logger().info(
            f"[LoRA] rank={cfg.lora_rank} alpha={lora_alpha} "
            f"scale={lora_alpha / cfg.lora_rank:.2f} adapters={lora / 1e6:.2f}M "
            f"trainable={trainable / 1e6:.2f}M total={total / 1e6:.2f}M "
            f"({100.0 * trainable / max(total, 1):.2f}% trainable)"
        )
        assert lora > 0, (
            "is_lora=True but no LoRA parameters were created; check that "
            "LoraConfig.target_modules matches this model's module names."
        )

    return model


def tag_vlm_subtree(model, is_vlm: bool):
    for n, m in model.named_modules():
        setattr(m, "_to_lora", is_vlm)

# ruff: noqa: B006

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import distribute_tensor

from weathergen.common.config import Config, get_path_model, merge_configs
from weathergen.model.layers import MLP
from weathergen.model.model import Model, ModelParams
from weathergen.model.utils import apply_fct_to_blocks, freeze_weights
from weathergen.utils.distributed import is_root
from weathergen.utils.performance import register_nvtx_hooks
from weathergen.utils.utils import get_dtype

logger = logging.getLogger(__name__)


# same as in config: student_teacher, forecasting, masking
type TrainingMode = str


class _AttnMlpBlock(torch.nn.Module):
    """Groups consecutive layers (typically Attention + MLP) into one FSDP unit."""

    def __init__(self, layers: list[torch.nn.Module]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x


def _replace_module_list(
    module_list: torch.nn.ModuleList, new_blocks: list[torch.nn.Module]
) -> None:
    """Replace ModuleList contents in-place while preserving the attribute binding."""
    while len(module_list) > 0:
        module_list.pop(0)
    for block in new_blocks:
        module_list.append(block)


def _shard_module_list_as_blocks(
    module_list: torch.nn.ModuleList,
    fsdp_kwargs: dict,
    *,
    reshard_after_forward: bool = True,
) -> int:
    """Pair flat [Attn, MLP, (LayerNorm), ...] into blocks and ``fully_shard`` each pair.

    Returns the number of FSDP units created.
    """
    children = list(module_list)
    new_blocks: list[torch.nn.Module] = []
    n_sharded = 0
    i = 0
    while i < len(children):
        child = children[i]
        if isinstance(child, torch.nn.LayerNorm):
            # Affine-free LayerNorm has no parameters; keep as-is.
            new_blocks.append(child)
            i += 1
            continue

        shard_kwargs = dict(fsdp_kwargs)
        if not reshard_after_forward:
            shard_kwargs["reshard_after_forward"] = False

        if i + 1 < len(children) and isinstance(children[i + 1], MLP):
            block = _AttnMlpBlock([children[i], children[i + 1]])
            fully_shard(block, **shard_kwargs)
            new_blocks.append(block)
            n_sharded += 1
            i += 2
        else:
            fully_shard(child, **shard_kwargs)
            new_blocks.append(child)
            n_sharded += 1
            i += 1

    _replace_module_list(module_list, new_blocks)
    return n_sharded


def _hsdp_shard_size(cf) -> int | None:
    """Resolve HSDP shard group size from config, or None for full-world FSDP.

    Config ``hsdp_shard_size``:
      - null / false / 0: classic FSDP (AllGather over world_size)
      - true: auto = world_size // num_nodes (GPUs per node under SLURM)
      - int: explicit shard size (must divide world_size), e.g. 4 on Jupiter
    """
    val = cf.get("hsdp_shard_size", None)
    if val is None or val is False or val == 0:
        return None
    if val is True:
        world_size = dist.get_world_size()
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
        if world_size % num_nodes != 0:
            raise ValueError(
                f"hsdp_shard_size=true requires world_size ({world_size}) divisible by "
                f"SLURM_JOB_NUM_NODES ({num_nodes})."
            )
        return world_size // num_nodes
    return int(val)


def _fsdp_device_mesh(cf):
    """Build a 2D (replicate, shard) mesh for HSDP, or None for default 1D FSDP."""
    shard_size = _hsdp_shard_size(cf)
    if shard_size is None:
        return None
    world_size = dist.get_world_size()
    if shard_size < 1 or world_size % shard_size != 0:
        raise ValueError(
            f"hsdp_shard_size={shard_size} must be >= 1 and divide world_size={world_size}."
        )
    replicate_size = world_size // shard_size
    mesh = init_device_mesh(
        "cuda",
        (replicate_size, shard_size),
        mesh_dim_names=("replicate", "shard"),
    )
    if is_root():
        logger.info(
            "HSDP DeviceMesh: replicate=%d × shard=%d (AllGather group size %d)",
            replicate_size,
            shard_size,
            shard_size,
        )
    return mesh


def init_model_and_shard(
    cf,
    dataset,
    run_id_contd,
    mini_epoch_contd,
    training_mode,
    device,
    with_ddp,
    with_fsdp,
    overrides={},
):
    model_creation_device = "meta" if with_ddp and with_fsdp else "cuda"
    with torch.device(model_creation_device):
        model = get_model(cf, training_mode, dataset, overrides)

    if cf.get("profiling", {}).get("nvtx_annotate", False):
        logger.info("Registering NVTX hooks for model.")
        register_nvtx_hooks(model)

    # freeze request model part
    apply_fct_to_blocks(model, cf.freeze_modules, freeze_weights)

    # TODO: this should be handled in the encoder to be close where q_cells is defined
    if "q_cells" in cf.freeze_modules:
        model.encoder.q_cells.requires_grad = False

    mesh = None
    if with_ddp and not with_fsdp:
        # create DDP model if running without FSDP
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            broadcast_buffers=True,
            find_unused_parameters=cf.get("ddp_find_unused_parameters", True),
            gradient_as_bucket_view=True,
            bucket_cap_mb=512,
        )

    elif with_ddp and with_fsdp:
        # with DDP *and() FSDP (optional 2D HSDP mesh via hsdp_shard_size)
        # Block-level sharding: pair Attention+MLP into one fully_shard unit to cut
        # AllGather/AllReduce count vs wrapping every leaf module.
        mesh = _fsdp_device_mesh(cf)
        fsdp_kwargs = {
            "mp_policy": (
                MixedPrecisionPolicy(
                    param_dtype=get_dtype(cf.mixed_precision_dtype),
                    reduce_dtype=torch.float32,
                )
                if cf.with_mixed_precision
                else None
            ),
        }
        if mesh is not None:
            fsdp_kwargs["mesh"] = mesh

        n_fsdp_units = 0
        n_fsdp_units += _shard_module_list_as_blocks(
            model.encoder.ae_local_engine.ae_local_blocks, fsdp_kwargs
        )
        if hasattr(model.encoder.ae_local_global_engine, "ae_adapter"):
            n_fsdp_units += _shard_module_list_as_blocks(
                model.encoder.ae_local_global_engine.ae_adapter, fsdp_kwargs
            )
        elif hasattr(model.encoder.ae_local_global_engine, "mlp_blocks"):
            n_fsdp_units += _shard_module_list_as_blocks(
                model.encoder.ae_local_global_engine.mlp_blocks, fsdp_kwargs
            )
        n_fsdp_units += _shard_module_list_as_blocks(
            model.encoder.ae_global_engine.ae_global_blocks, fsdp_kwargs
        )
        n_fsdp_units += _shard_module_list_as_blocks(
            model.forecast_engine.fe_blocks,
            fsdp_kwargs,
            # Keep FE params unsharded after forward for multi-step rollout / pushforward.
            reshard_after_forward=False,
        )

        for head in model.latent_heads.values():
            fully_shard(head, **fsdp_kwargs)
            n_fsdp_units += 1

        full_precision_fsdp_kwargs = {
            "mp_policy": (
                MixedPrecisionPolicy(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                )
                if cf.with_mixed_precision
                else None
            ),
        }
        if mesh is not None:
            full_precision_fsdp_kwargs["mesh"] = mesh

        for tte in model.target_token_engines.values():
            fully_shard(tte, **full_precision_fsdp_kwargs)
            n_fsdp_units += 1

        if is_root():
            logger.info("FSDP block-level sharding: %d units (Attention+MLP pairs / heads)", n_fsdp_units)

    if with_ddp and with_fsdp:
        fully_shard(model, **({"mesh": mesh} if mesh is not None else {}))
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            assert tensor.device == torch.device("meta")

        # For reasons we do not yet fully understand, when using train continue in some
        # instances, FSDP2 does not register the forward_channels and forward_columns
        # functions in the embedding engine as forward functions. Thus, yielding a crash
        # because the input tensors are not converted to DTensors. This seems to primarily
        # occur during validation.
        for embed in model.encoder.embed_engine.embeds.values():
            torch.distributed.fsdp.register_fsdp_forward_method(embed, "forward")

    # complete initalization and load model if inference/continuing a run
    if run_id_contd is not None:
        if is_root():
            logger.info(f"Continuing run with id={run_id_contd} at mini_epoch {mini_epoch_contd}.")
        model = load_model(cf, model, device, run_id_contd, mini_epoch_contd)
    elif cf.get("load_chkpt", {}).get("run_id", None):
        run_id = cf.load_chkpt.run_id
        mini_epoch = cf.load_chkpt.get("mini_epoch", -1)
        if is_root():
            logger.info(f"Loading checkpoint from id={run_id} at mini_epoch {mini_epoch}.")
        model = load_model(cf, model, device, run_id, mini_epoch)
    else:
        if with_ddp and with_fsdp:
            model.to_empty(device="cuda")
            if with_fsdp:
                model.reset_parameters()

    # model params
    model_params = ModelParams(cf).create(cf)
    model_params.reset_parameters(cf)
    model_params = model_params.to(f"cuda:{cf.local_rank}")

    return model, model_params


def load_model(cf, model, device, run_id: str, mini_epoch=-1):
    """Loads model state from checkpoint and checks for missing and unused keys.
    Args:
        run_id : model_id of the trained model
        mini_epoch : The mini_epoch to load. Default (-1) is the latest mini_epoch
    """

    path_run = get_path_model(run_id=run_id)
    mini_epoch_id = (
        f"chkpt{mini_epoch:05d}" if mini_epoch != -1 and mini_epoch is not None else "latest"
    )
    filename = f"{run_id}_{mini_epoch_id}.chkpt"

    params = torch.load(
        path_run / filename, map_location=torch.device("cpu"), mmap=True, weights_only=True
    )

    is_model_sharded = cf.with_ddp and cf.with_fsdp
    if is_model_sharded:
        meta_sharded_sd = model.state_dict()
        maybe_sharded_sd = {}
        for param_name, full_tensor in params.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            if sharded_meta_param is None:
                logger.warning(f"Parameter {param_name} from checkpoint not found in model.")
                continue
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
            # maybe_sharded_sd[param_name.replace("module.", "")] = nn.Parameter(sharded_tensor)
            maybe_sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
        # choose `assign=True` for sharded model since we cannot call `copy_` on meta tensor
        mkeys, ukeys = model.load_state_dict(maybe_sharded_sd, strict=False, assign=True)

        # new network parts (e.g. for fine-tuning)
        if mkeys:
            # Get the unique parent modules for the missing parameters
            new_modules_to_init = {key.rsplit(".", 1)[0] for key in mkeys}

            # Find the highest-level "root" new modules to avoid redundant initializations
            root_new_modules = set()
            for path in sorted(list(new_modules_to_init)):
                if not any(path.startswith(root + ".") for root in root_new_modules):
                    root_new_modules.add(path)

            # Get all modules for quick lookup and initialize the new ones
            all_modules = dict(model.named_modules())
            for path in root_new_modules:
                if is_root():
                    logger.info(f"Initializing new module not found in checkpoint: {path}")
                module_to_init = all_modules[path]
                module_to_init.to_empty(device="cuda")
                module_to_init.reset_parameters()

    else:
        # fix mismatch between state_dict keys that can occur between interactive/non-interactive
        model_has_prefix_module = list(model.state_dict().keys())[0].split(".")[0] == "module"
        params_has_prefix_module = list(params.keys())[0].split(".")[0] == "module"
        if model_has_prefix_module and not params_has_prefix_module:
            # add "module." prefix
            params_temp = {}
            for k in params.keys():
                params_temp["module." + k] = params[k]
            params = params_temp
        elif not model_has_prefix_module and params_has_prefix_module:
            # remove "module." prefix
            params_temp = {}
            for k in params.keys():
                params_temp[k.replace("module.", "")] = params[k]
            params = params_temp
        # load checkpoint
        mkeys, ukeys = model.load_state_dict(params, strict=False)
        model = model.to(device)

    # warn about difference in checkpoint and model
    if len(mkeys) == 0 and len(ukeys) == 0:
        logger.info(f"Checkpoint {filename} loaded successfully with all weights matching.")
    if len(mkeys) > 0:
        logger.warning(f"Missing keys when loading model: {mkeys}")
    if len(ukeys) > 0:
        logger.warning(f"Unused keys when loading model: {ukeys}")

    return model


def get_model(cf: Config, training_mode: TrainingMode, dataset, overrides):
    """
    Create model

    cf :
    training_mode :
    dataset :
    """

    # TODO: how to avoid the dependence on dataset
    sources_size = dataset.get_sources_size()
    targets_num_channels = dataset.get_targets_num_channels()
    targets_coords_size = dataset.get_targets_coords_size()

    cf_with_overrides = merge_configs(cf, overrides)
    return Model(
        cf_with_overrides, sources_size, targets_num_channels, targets_coords_size
    ).create()

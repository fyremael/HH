from .attention import HouseholderSelfAttention
from .core import (
    BlockDiagonalRoPECore,
    HouseholderRoPE,
    HouseholderRoPEConfig,
    apply_householder_rope,
    apply_householder_stack,
    householder_normalize,
    materialize_Q,
    premix_qk,
)
from .diagnostics import (
    attention_logit_path_error,
    block_mixing_energy,
    commutator_defect,
    orthogonality_defect,
    reflector_utilization,
    relativity_defect,
    reversibility_defect,
    summarize_householder_rope_diagnostics,
)

__all__ = [
    "BlockDiagonalRoPECore",
    "HouseholderRoPE",
    "HouseholderRoPEConfig",
    "HouseholderSelfAttention",
    "apply_householder_rope",
    "apply_householder_stack",
    "attention_logit_path_error",
    "block_mixing_energy",
    "commutator_defect",
    "householder_normalize",
    "materialize_Q",
    "orthogonality_defect",
    "premix_qk",
    "reflector_utilization",
    "relativity_defect",
    "reversibility_defect",
    "summarize_householder_rope_diagnostics",
]

try:
    from .jax_attention import (
        householder_attention as jax_householder_attention,
        scaled_dot_product_attention as jax_scaled_dot_product_attention,
    )
    from .jax_core import (
        JaxBlockDiagonalRoPECore,
        JaxHouseholderRoPE,
        JaxHouseholderRoPEConfig,
        apply_householder_rope as jax_apply_householder_rope,
        apply_householder_stack as jax_apply_householder_stack,
        householder_normalize as jax_householder_normalize,
        materialize_Q as jax_materialize_Q,
        premix_qk as jax_premix_qk,
    )
    from .jax_diagnostics import (
        attention_logit_path_error as jax_attention_logit_path_error,
        block_mixing_energy as jax_block_mixing_energy,
        commutator_defect as jax_commutator_defect,
        orthogonality_defect as jax_orthogonality_defect,
        reflector_utilization as jax_reflector_utilization,
        relativity_defect as jax_relativity_defect,
        reversibility_defect as jax_reversibility_defect,
        summarize_householder_rope_diagnostics as summarize_householder_rope_jax_diagnostics,
    )
except Exception:
    pass
else:
    __all__.extend(
        [
            "JaxBlockDiagonalRoPECore",
            "JaxHouseholderRoPE",
            "JaxHouseholderRoPEConfig",
            "jax_apply_householder_rope",
            "jax_apply_householder_stack",
            "jax_attention_logit_path_error",
            "jax_block_mixing_energy",
            "jax_commutator_defect",
            "jax_householder_attention",
            "jax_householder_normalize",
            "jax_materialize_Q",
            "jax_orthogonality_defect",
            "jax_premix_qk",
            "jax_reflector_utilization",
            "jax_relativity_defect",
            "jax_reversibility_defect",
            "jax_scaled_dot_product_attention",
            "summarize_householder_rope_jax_diagnostics",
        ]
    )

try:
    from .flax_modules import FlaxHouseholderRoPE, FlaxHouseholderSelfAttention
except Exception:
    pass
else:
    __all__.extend(
        [
            "FlaxHouseholderRoPE",
            "FlaxHouseholderSelfAttention",
        ]
    )

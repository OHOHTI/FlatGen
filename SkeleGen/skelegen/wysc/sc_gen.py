"""
sc_gen.py — Combined Wyckoff + Sublattice Constrained Generation

Generates crystal structures using DiffCSP++ with:
  1. Wyckoff symmetry constraints (ops projection + affine enforcement)
  2. Fixed sublattice atom positions and types (mask replacement)
  3. Fixed lattice parameters (forward-diffused known lattice)

Constraints dataset is generated on-the-fly via run_pipeline() (DP sampler)
or run_pipeline_template() (template-based) depending on --use_template.

Output: .pt checkpoint files with generated structures

Usage:
    python -m wysc.sc_gen --model_path <path> --num_batches 10
    python -m wysc.sc_gen --model_path <path> --num_batches 10 --use_template
"""

from __future__ import annotations

import gc
import os
import sys
import time
import argparse
from copy import deepcopy as dc
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_utils import load_model, lattices_to_params_shape
from wysc.build_dataset import (
    DEFAULT_NON_TRICLINIC_WEIGHT,
    DEFAULT_SKELETON_SYMMETRY_POWER,
    run_pipeline,
)
from wysc.build_dataset_template import run_pipeline_template

MAX_ATOMIC_NUM = 100


# ── sample_wysc: core sampling function (monkey-patched onto model) ─────────

@torch.no_grad()
def sample_wysc(self, batch, step_lr=1e-5, t_stop_x=0.0, t_stop_l=0.0):
    """
    DiffCSP++ (w/ type diffusion) denoising with sublattice mask replacement.

    Coordinates: Wyckoff-projected denoising + mask replacement for constrained atoms.
    Lattice: forward-diffused known lattice (constrained) or standard denoising (unconstrained).
    Atom types: continuous (N, 100) type diffusion with mask replacement for constrained atoms.

    Parameters
    ----------
    batch : PyG Batch with fields from WyckoffSubLatDataset
    step_lr : Langevin step size
    t_stop_x : fraction of T below which coord/type constraints are disabled
    t_stop_l : fraction of T below which lattice constraints are disabled
    """
    batch_size = batch.num_graphs
    device = self.device

    # ── Known values ─────────────────────────────────────────────────────
    x_0_known = batch.frac_coords  # (N_total, 3) — known values in [:num_known], zeros elsewhere
    t_0_known = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()  # (N, 100)

    mask_x = batch.is_constrained.float().unsqueeze(-1).to(device)  # (N, 1) — for coords
    mask_x_inv = 1.0 - mask_x
    mask_t = batch.is_constrained.float().unsqueeze(-1).to(device)  # (N, 1) — for types (always on, no adaptive unmasking)
    mask_t_inv = 1.0 - mask_t

    # Known lattice → crystal family representation
    lattice_known = batch.lattice_known  # (B, 3, 3)
    crys_fam_0 = self.crystal_family.de_so3(lattice_known)
    crys_fam_0 = self.crystal_family.m2v(crys_fam_0)
    crys_fam_0 = self.crystal_family.proj_k_to_spacegroup(crys_fam_0, batch.spacegroup)

    # ── Initialize at t=T ────────────────────────────────────────────────
    x_T = torch.rand([batch.num_nodes, 3]).to(device)
    crys_fam_T = torch.randn([batch_size, 6]).to(device)
    crys_fam_T = self.crystal_family.proj_k_to_spacegroup(crys_fam_T, batch.spacegroup)
    t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(device)

    # Wyckoff affine enforcement on initial x
    x_T_all = torch.cat(
        [x_T[batch.anchor_index],
         torch.ones(batch.ops.size(0), 1).to(device)],
        dim=-1,
    ).unsqueeze(-1)
    x_T = (batch.ops @ x_T_all).squeeze(-1)[:, :3] % 1.0

    # Anchor projection on initial types
    t_T = t_T[batch.anchor_index]

    l_T = self.crystal_family.v2m(crys_fam_T)

    time_start = self.beta_scheduler.timesteps - 1
    T = self.beta_scheduler.timesteps

    # Only keep the current timestep to avoid OOM from storing all 999 steps
    cur = {
        'num_atoms': batch.num_atoms,
        'atom_types': t_T,
        'frac_coords': x_T % 1.0,
        'lattices': l_T,
        'crys_fam': crys_fam_T,
    }

    for t in tqdm(range(time_start, 0, -1)):

        # ── Adaptive unmasking ───────────────────────────────────────────
        if t < t_stop_x * T:
            mask_x = torch.zeros_like(mask_x)
            mask_x_inv = 1.0 - mask_x

        lattice_constrained = (t >= t_stop_l * T)

        # ── Scheduler values ─────────────────────────────────────────────
        times = torch.full((batch_size,), t, device=device)
        time_emb = self.time_embedding(times)

        alphas = self.beta_scheduler.alphas[t]
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
        sigmas = self.beta_scheduler.sigmas[t]
        sigma_x = self.sigma_scheduler.sigmas[t]
        sigma_norm = self.sigma_scheduler.sigmas_norm[t]

        c0 = 1.0 / torch.sqrt(alphas)
        c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

        # Forward-diffuse coefficients for known values at time t
        C0_t = torch.sqrt(alphas_cumprod)
        C1_t = torch.sqrt(1.0 - alphas_cumprod)

        # Forward-diffuse coefficients at t-1
        alphas_cumprod_m1 = self.beta_scheduler.alphas_cumprod[t - 1]
        C0_m1 = torch.sqrt(alphas_cumprod_m1)
        C1_m1 = torch.sqrt(1.0 - alphas_cumprod_m1)
        sigma_x_m1 = self.sigma_scheduler.sigmas[t - 1]

        x_t = cur['frac_coords']
        l_t = cur['lattices']
        crys_fam_t = cur['crys_fam']
        t_t = cur['atom_types']  # (N, 100) continuous

        # ==============================================================
        # CORRECTOR  (types unchanged, same as diffusion_w_type.py)
        # ==============================================================

        # 1. Wyckoff-projected noise for coord update
        rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
        std_x = torch.sqrt(2 * step_size)

        rand_x_anchor = rand_x[batch.anchor_index]
        rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

        # 2. Decoder forward pass (takes continuous type probs)
        pred_crys_fam, pred_x, pred_t = self.decoder(
            time_emb, t_t, x_t, crys_fam_t,
            batch.num_atoms, batch.batch,
        )
        pred_x = pred_x * torch.sqrt(sigma_norm)

        # 3. Wyckoff-project coord prediction
        pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
        pred_x_anchor = scatter(
            pred_x_proj, batch.anchor_index, dim=0, reduce='mean',
        )[batch.anchor_index]
        pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

        # 4. Langevin update → x_denoised
        x_t_minus_05_unk = x_t - step_size * pred_x + std_x * rand_x

        # 5-6. Forward-diffuse known coords with FRESH Wyckoff-projected noise
        rand_x_known = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        rand_x_known_anchor = rand_x_known[batch.anchor_index]
        rand_x_known_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_known_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x_known = (batch.ops[:, :3, :3] @ rand_x_known_anchor.unsqueeze(-1)).squeeze(-1)
        x_t_minus_05_known = (x_0_known + sigma_x * rand_x_known) % 1.0

        # 7. Mask replacement for coords
        x_t_minus_05 = mask_x * x_t_minus_05_known + mask_x_inv * x_t_minus_05_unk

        # 8. Wyckoff affine enforcement
        frac_all = torch.cat(
            [x_t_minus_05[batch.anchor_index],
             torch.ones(batch.ops.size(0), 1).to(device)],
            dim=-1,
        ).unsqueeze(-1)
        x_t_minus_05 = (batch.ops @ frac_all).squeeze(-1)[:, :3] % 1.0

        # Corrector: types unchanged
        t_t_minus_05 = t_t

        # Corrector: lattice unchanged (standard DiffCSP++ pattern)
        crys_fam_t_minus_05 = crys_fam_t

        # If lattice constrained, forward-diffuse known lattice for decoder input
        if lattice_constrained:
            rand_l_known = torch.randn_like(crys_fam_0) if t > 1 else torch.zeros_like(crys_fam_0)
            rand_l_known = self.crystal_family.proj_k_to_spacegroup(rand_l_known, batch.spacegroup)
            crys_fam_t_minus_05 = C0_t * crys_fam_0 + C1_t * rand_l_known

        # ==============================================================
        # PREDICTOR
        # ==============================================================

        # Coord noise (Wyckoff-projected)
        rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
        step_size_pred = (sigma_x ** 2 - adjacent_sigma_x ** 2)
        std_x_pred = torch.sqrt(
            (adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2)
        )

        rand_x_anchor = rand_x[batch.anchor_index]
        rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

        # Type noise (anchor-projected)
        rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
        rand_t = rand_t[batch.anchor_index]

        # Lattice noise
        rand_crys_fam = torch.randn_like(crys_fam_T)
        rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)

        # Decoder forward pass
        pred_crys_fam, pred_x, pred_t = self.decoder(
            time_emb, t_t_minus_05, x_t_minus_05, crys_fam_t_minus_05,
            batch.num_atoms, batch.batch,
        )
        pred_x = pred_x * torch.sqrt(sigma_norm)

        # Wyckoff-project coord prediction
        pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
        pred_x_anchor = scatter(
            pred_x_proj, batch.anchor_index, dim=0, reduce='mean',
        )[batch.anchor_index]
        pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

        # Wyckoff-project type prediction
        pred_t = scatter(pred_t, batch.anchor_index, dim=0, reduce='mean')[batch.anchor_index]

        # ── Coord predictor update + mask replacement ────────────────────
        x_t_minus_1_unk = x_t_minus_05 - step_size_pred * pred_x + std_x_pred * rand_x

        rand_x_known = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        rand_x_known_anchor = rand_x_known[batch.anchor_index]
        rand_x_known_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_known_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x_known = (batch.ops[:, :3, :3] @ rand_x_known_anchor.unsqueeze(-1)).squeeze(-1)
        x_t_minus_1_known = (x_0_known + sigma_x_m1 * rand_x_known) % 1.0

        x_t_minus_1 = mask_x * x_t_minus_1_known + mask_x_inv * x_t_minus_1_unk

        # Wyckoff affine enforcement
        frac_all = torch.cat(
            [x_t_minus_1[batch.anchor_index],
             torch.ones(batch.ops.size(0), 1).to(device)],
            dim=-1,
        ).unsqueeze(-1)
        x_t_minus_1 = (batch.ops @ frac_all).squeeze(-1)[:, :3] % 1.0

        # ── Type predictor update + mask replacement ─────────────────────
        t_t_minus_1_unk = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

        # Forward-diffuse known types at t-1 noise level
        rand_t_known = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
        rand_t_known = rand_t_known[batch.anchor_index]
        t_t_minus_1_known = C0_m1 * t_0_known + C1_m1 * rand_t_known

        t_t_minus_1 = mask_t * t_t_minus_1_known + mask_t_inv * t_t_minus_1_unk

        # Anchor projection (ensures orbit consistency)
        t_t_minus_1 = t_t_minus_1[batch.anchor_index]

        # ── Lattice predictor ────────────────────────────────────────────
        if lattice_constrained:
            rand_l_known = torch.randn_like(crys_fam_0) if t > 1 else torch.zeros_like(crys_fam_0)
            rand_l_known = self.crystal_family.proj_k_to_spacegroup(rand_l_known, batch.spacegroup)
            crys_fam_t_minus_1 = C0_m1 * crys_fam_0 + C1_m1 * rand_l_known
        else:
            crys_fam_t_minus_1 = c0 * (crys_fam_t_minus_05 - c1 * pred_crys_fam) + sigmas * rand_crys_fam
            crys_fam_t_minus_1 = self.crystal_family.proj_k_to_spacegroup(
                crys_fam_t_minus_1, batch.spacegroup,
            )

        l_t_minus_1 = self.crystal_family.v2m(crys_fam_t_minus_1)

        cur = {
            'num_atoms': batch.num_atoms,
            'atom_types': t_t_minus_1,          # (N, 100) continuous
            'frac_coords': x_t_minus_1 % 1.0,
            'lattices': l_t_minus_1,
            'crys_fam': crys_fam_t_minus_1,
        }

    # Convert final types from continuous → integer
    res = dc(cur)
    res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1

    return res


# ── diffusion_wysc: batch loop with chunked saving ─────────────────────────

def diffusion_wysc(
    loader, model, step_lr, t_stop_x, t_stop_l,
    save_dir, save_every=100, use_template=False,
):
    """Run constrained diffusion and save results in chunks."""
    os.makedirs(save_dir, exist_ok=True)

    frac_coords = []
    num_atoms = []
    num_known = []
    atom_types = []
    lattices = []
    sublat_ids = []
    file_idx = 0
    total_batches = len(loader)
    fname_prefix = "eval_gen_wysc_tmpl" if use_template else "eval_gen_wysc"

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()

        outputs = model.sample_wysc(
            batch, step_lr=step_lr, t_stop_x=t_stop_x, t_stop_l=t_stop_l,
        )

        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

        # Collect num_known per graph
        nk = batch.num_known
        if isinstance(nk, int):
            nk = torch.LongTensor([nk])
        elif isinstance(nk, torch.Tensor):
            nk = nk.detach().cpu().reshape(-1)
        num_known.append(nk)

        # Collect sublat_ids
        if hasattr(batch, 'sublat_id'):
            sid = batch.sublat_id
            if isinstance(sid, list):
                sublat_ids.extend(sid)
            else:
                sublat_ids.append(sid)

        print(f'batch {idx + 1}/{total_batches} done')

        # Save checkpoint every save_every batches or at the last batch
        if (idx + 1) % save_every == 0 or (idx + 1) == total_batches:
            fc = torch.cat(frac_coords, dim=0)
            na = torch.cat(num_atoms, dim=0)
            nk_cat = torch.cat(num_known, dim=0)
            at = torch.cat(atom_types, dim=0)
            lt = torch.cat(lattices, dim=0)
            le, an = lattices_to_params_shape(lt)

            chunk = {
                'eval_setting': None,
                'num_atoms': na,
                'num_known': nk_cat,
                'frac_coords': fc,
                'atom_types': at,
                'lengths': le,
                'angles': an,
                'sublat_ids': sublat_ids[:],
            }
            out_path = os.path.join(save_dir, f'{fname_prefix}_{file_idx:04d}.pt')
            torch.save(chunk, out_path)
            print(f'Saved {out_path} ({len(na)} materials)')
            file_idx += 1

            # Clear accumulators
            frac_coords, num_atoms, num_known, atom_types, lattices, sublat_ids = (
                [], [], [], [], [], [],
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wyckoff + sublattice constrained crystal generation via DiffCSP++.",
    )
    parser.add_argument("--model_path", required=True, help="Path to trained DiffCSP++ model directory")
    parser.add_argument("--step_lr", type=float, default=1e-5)
    parser.add_argument("--t_stop_x", type=float, default=0.0,
                        help="Fraction of T below which coord constraints are disabled (0=always on)")
    parser.add_argument("--t_stop_l", type=float, default=0.0,
                        help="Fraction of T below which lattice constraints are disabled (0=always on)")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_batches", type=int, required=True,
                        help="Number of batches to run (total samples = batch_size × num_batches)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory for output chunks (default: <model_path>/wysc_gen/ or wysc_tmpl_gen/)")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N batches")
    parser.add_argument("--natm_min", type=int, default=5)
    parser.add_argument("--natm_max", type=int, default=20)

    # Mode selection
    parser.add_argument("--use_template", action="store_true",
                        help="Use template-based Wyckoff sampling (step3 templates) instead of DP sampler")

    # DP sampler parameters (used when --use_template is not set)
    parser.add_argument("--candidates", type=str, default=None,
                        help="Path to step2_candidates.pkl (default: data/step2_candidates.pkl)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="DP sampler temperature (>1 = more diverse, <1 = sharper)")
    parser.add_argument("--non_triclinic_weight", type=float, default=DEFAULT_NON_TRICLINIC_WEIGHT,
                        help="Relative weight for non-triclinic (SG>2) vs triclinic g_full assignments. "
                             "Default 5.0; set 1.0 for uniform weighting.")
    parser.add_argument("--skeleton_symmetry_power", type=float, default=DEFAULT_SKELETON_SYMMETRY_POWER,
                        help="Exponent on max-SG for skeleton sampling weight. "
                             "Default 1.2; set 0.0 for uniform skeleton sampling.")

    # Template parameters (used when --use_template is set)
    parser.add_argument("--templates", type=str, default=None,
                        help="Path to step3_compatible_templates.pkl "
                             "(default: data/step3_compatible_templates.pkl)")
    parser.add_argument("--precompute_cache", type=str, default=None,
                        help="Path to cache precomputed template assignments "
                             "(default: data/precomputed_template_assignments.pkl)")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load model ───────────────────────────────────────────────────────
    model_path = Path(args.model_path)
    model, _, cfg = load_model(model_path, load_data=False)
    if torch.cuda.is_available():
        model.to('cuda')

    # Monkey-patch the constrained sampling function
    model.sample_wysc = sample_wysc.__get__(model)

    # ── Generate constraints dataset on-the-fly ──────────────────────────
    total_samples = args.batch_size * args.num_batches
    natm_range = (args.natm_min, args.natm_max)

    if args.use_template:
        print(f"Generating {total_samples} template-based samples "
              f"({args.num_batches} batches × {args.batch_size})...")
        pipeline_kwargs = dict(
            natm_range=natm_range,
            total_num=total_samples,
            seed=args.seed,
        )
        if args.templates:
            pipeline_kwargs["templates_pkl"] = args.templates
        if args.precompute_cache:
            pipeline_kwargs["precompute_cache"] = args.precompute_cache
        dataset = run_pipeline_template(**pipeline_kwargs)
        default_subdir = "wysc_tmpl_gen"
    else:
        print(f"Generating {total_samples} constraint samples "
              f"({args.num_batches} batches × {args.batch_size})...")
        pipeline_kwargs = dict(
            total_num=total_samples,
            seed=args.seed,
            natm_range=natm_range,
            temperature=args.temperature,
            non_triclinic_weight=args.non_triclinic_weight,
            skeleton_symmetry_power=args.skeleton_symmetry_power,
        )
        if args.candidates:
            pipeline_kwargs["candidates_pkl"] = args.candidates
        dataset = run_pipeline(**pipeline_kwargs)
        default_subdir = "wysc_gen"

    loader = DataLoader(dataset, batch_size=args.batch_size)

    # ── Output directory ─────────────────────────────────────────────────
    save_dir = args.save_dir or str(model_path / default_subdir / "chunks")

    print(f"Model: {model_path}")
    print(f"Dataset: {len(dataset)} samples")
    print(f"step_lr={args.step_lr}, t_stop_x={args.t_stop_x}, t_stop_l={args.t_stop_l}")
    if args.use_template:
        print(f"templates={args.templates or 'data/step3_compatible_templates.pkl (default)'}")
    else:
        print(f"dp temperature={args.temperature}, "
              f"non_triclinic_weight={pipeline_kwargs['non_triclinic_weight']}, "
              f"skeleton_symmetry_power={pipeline_kwargs['skeleton_symmetry_power']}")
    print(f"Save dir: {save_dir}")
    print()

    t0 = time.time()
    diffusion_wysc(
        loader, model, args.step_lr,
        t_stop_x=args.t_stop_x,
        t_stop_l=args.t_stop_l,
        save_dir=save_dir,
        save_every=args.save_every,
        use_template=args.use_template,
    )
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

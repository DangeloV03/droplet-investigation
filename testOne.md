# HeteroNVTDrivenChain Validation Results

All tests run on a 64×64 periodic lattice with density 0.025 (102 particles total),
1000 time units equilibration, 100 chunks × 100 time units production. β = 1, η = 1, Λ = 100.

---

## Equilibrium tests (Δμ = 0, bond_energy = 0)

Expected B/I ratio: e^{−βΔf}

| Δf   | Expected | Measured | Error |
|------|----------|----------|-------|
| +0.1 | 0.9048   | 0.9122   | 0.82% |
| −0.1 | 1.1052   | 1.1199   | 1.33% |

Both pass. Sign flip of Δf correctly inverts the ratio.

---

## Nonequilibrium tests (bond_energy = 0)

Expected B/I ratio from steady-state rate balance: k_{I→B} / k_{B→I}.

| Δf   | Δμ   | Expected | Measured | Error |
|------|------|----------|----------|-------|
| +0.1 | +0.5 | 0.7357   | 0.7540   | 2.48% |
| +0.1 | −0.5 | 1.1264   | 1.1635   | 3.30% |

Both pass. Positive Δμ drives the system toward I (ratio < equilibrium), negative Δμ drives toward B (ratio > equilibrium), as expected from stochastic thermodynamics.

---

## Bond energy tests (Δμ = 0, Δf = 0.1)

| bond_energy | Regime            | Result |
|-------------|-------------------|--------|
| −1.0        | Disordered (above critical point βε_c ≈ −1.76) | Mean B/I = 1.018, homogeneous lattice confirmed visually ✓ |
| −2.95       | Phase-separated (below critical point) | B-rich droplet visible, mean global B/I = 6.31 (dominated by condensate composition, not a rate-balance test) ✓ |

---

## Summary

The chain correctly reproduces equilibrium Boltzmann statistics, responds to Δμ in both
directions, and phase-separates at strong coupling. All quantitative tests pass within
expected finite-size noise (~1–3% with N = 102 particles).
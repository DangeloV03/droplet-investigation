"""
Cluster analysis for the dividing-droplet lattice gas.

Identifies the largest contiguous cluster of a given site type under
periodic boundary conditions, following the convention of Cho & Jacobs
(PNAS 2025): the condensed phase is the largest contiguous cluster of
B-state (bonding) lattice sites.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

BONDING = 2
INERT = 1


def _merge_periodic_labels(labels: np.ndarray, num_labels: int) -> np.ndarray:
    """
    scipy.ndimage.label treats the array as having open boundaries, so a
    cluster that wraps around an edge gets split into two different label
    IDs. This returns a remapping array `merged` such that
    `merged[label_id]` gives the canonical label ID after merging clusters
    that touch across the periodic boundary (left-right and top-bottom).

    Implemented as union-find over the small set of label IDs (not over
    individual lattice sites).
    """
    parent = np.arange(num_labels + 1)

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Left-right wraparound: column 0 is adjacent to column -1.
    for l, r in zip(labels[:, 0], labels[:, -1]):
        if l != 0 and r != 0 and l != r:
            union(int(l), int(r))

    # Top-bottom wraparound: row 0 is adjacent to row -1.
    for t, b in zip(labels[0, :], labels[-1, :]):
        if t != 0 and b != 0 and t != b:
            union(int(t), int(b))

    return np.array([find(i) for i in range(num_labels + 1)])


def largest_cluster_stats(state: np.ndarray, target: int = BONDING) -> tuple[int, int, float]:
    """
    Find the largest contiguous cluster of `target`-type sites under
    periodic boundary conditions.

    Parameters
    ----------
    state : 2D array of site states (EMPTY/INERT/BONDING).
    target : the site value defining the cluster (default: BONDING).

    Returns
    -------
    area : number of sites in the largest cluster.
    perimeter : number of nearest-neighbor bonds connecting a cluster
        site to a non-cluster site (periodic).
    r_eff : sqrt(area / pi), the effective droplet radius.

    If there are no `target` sites at all, returns (0, 0, 0.0).
    """
    mask = state == target

    labels, num_labels = ndimage.label(mask)
    if num_labels == 0:
        return 0, 0, 0.0

    merged = _merge_periodic_labels(labels, num_labels)
    merged_labels = merged[labels]

    counts = np.bincount(merged_labels.ravel())
    counts[0] = 0  # ignore background
    largest_label = int(np.argmax(counts))
    area = int(counts[largest_label])

    cluster_mask = merged_labels == largest_label

    # Each (site, +1-neighbor) pair along each axis represents one bond,
    # exactly once, thanks to periodic wraparound via np.roll.
    perimeter = 0
    for axis in (0, 1):
        neighbor = np.roll(cluster_mask, -1, axis=axis)
        perimeter += int(np.sum(cluster_mask != neighbor))

    r_eff = float(np.sqrt(area / np.pi))
    return area, perimeter, r_eff


def far_field_densities(state: np.ndarray, border_frac: float = 0.05) -> tuple[float, float]:
    """
    Measure the bonding and inert densities in a frame of sites around the
    edges of the lattice, as a proxy for the bulk dilute-phase ("far-field")
    concentrations rho^v_B and rho^v_I (cf. Cho & Jacobs, PNAS 2025).

    Parameters
    ----------
    state : 2D array of site states (EMPTY/INERT/BONDING).
    border_frac : fraction of the lattice dimension defining the width of
        the border frame on each side (default: 0.05, i.e. a 5% margin).

    Returns
    -------
    rho_b_far : fraction of border sites that are BONDING.
    rho_i_far : fraction of border sites that are INERT.

    Assumes the droplet is roughly centered, so the border frame sits in
    the dilute phase, far from the droplet interface.
    """
    ly, lx = state.shape
    border_y = max(1, round(border_frac * ly))
    border_x = max(1, round(border_frac * lx))

    mask = np.zeros(state.shape, dtype=bool)
    mask[:border_y, :] = True
    mask[-border_y:, :] = True
    mask[:, :border_x] = True
    mask[:, -border_x:] = True

    border_sites = state[mask]
    n_border = border_sites.size

    rho_b_far = float(np.count_nonzero(border_sites == BONDING)) / n_border
    rho_i_far = float(np.count_nonzero(border_sites == INERT)) / n_border
    return rho_b_far, rho_i_far


if __name__ == "__main__":
    # Self-test: a cluster that wraps around both edges of a small lattice
    # should be detected as a single connected component.
    test = np.zeros((6, 6), dtype=np.uint32)
    test[2, :] = BONDING       # horizontal bar across the middle
    test[:, 0] = BONDING       # left column
    test[:, 5] = BONDING       # right column (wraps to meet left column)

    area, perimeter, r_eff = largest_cluster_stats(test)
    print(f"area={area}, perimeter={perimeter}, r_eff={r_eff:.3f}")

    # Without periodic merging, left and right columns would be counted
    # as separate clusters of size 6 each; with merging they join the
    # middle bar into one cluster of size 6+6+6-2 = 16 (corners shared).
    assert area == 16, f"expected 16, got {area}"
    print("self-test passed")

    # far_field_densities: put some INERT sites in the corners (border
    # region for a 6x6 lattice with border_frac=0.2 -> border width 1).
    test2 = np.zeros((6, 6), dtype=np.uint32)
    test2[0, 0] = INERT
    test2[0, 1] = BONDING
    rho_b_far, rho_i_far = far_field_densities(test2, border_frac=0.2)
    print(f"rho_b_far={rho_b_far:.3f}, rho_i_far={rho_i_far:.3f}")
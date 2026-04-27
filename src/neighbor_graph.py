from typing import List, Optional, Tuple, Dict, Any, Union, Literal, Set, Callable, Iterable, Type
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.preprocessing import normalize
from config import Config
import logging as log
from utils import TAXONOMY_FEATURES

class NeighbourGraph:
    """
    Build neighbour lists and compute interpolation weights.

    Modes supported:
    - taxonomy-only: discrete taxonomic distance
    - embedding-only: continuous neighbors on embeddings
    - hybrid: taxonomy + embedding ranking

    Neighbor selection modes (cfg.neighbor_mode):
    - "threshold": select all neighbors within a distance threshold
    - "knn": select K nearest neighbors

    Provides functions to compute NW weights and LLR coefficients per node.
    """

    def __init__(self, cfg: Config, bins_df: pd.DataFrame):
        self.cfg = cfg
        self.bins = bins_df.copy().reset_index(drop=True)
        self.n_bins = len(self.bins)

        # taxonomy columns expected: species, genus, subfamily, family, order, class, phylum
        self.tax_levels = TAXONOMY_FEATURES

        # output structures
        self.neighbours: List[List[int]] = [[] for _ in range(self.n_bins)]
        self.distances: List[np.ndarray] = [np.array([]) for _ in range(self.n_bins)]
        
        if self.cfg.use_embedding:
            self.embeddings: Optional[np.ndarray] = bins_df.get("embedding", None)
            self.bins_with_embedding: np.ndarray = bins_df.get("has_embedding", np.ones(self.n_bins, dtype=bool)).values


    def build_taxonomy_neighbors_knn(self, K: int) -> None:
        """
        Populate neighbours using taxonomic discrete distance with KNN.
        Selects the K nearest neighbors by taxonomy.
        
        Args:
            K: Number of nearest neighbors to select.
        """
        from tqdm import tqdm
        if "bin_uri" not in self.bins.columns:
            raise ValueError("bins dataframe must contain 'bin_uri'")
        
        log.debug(f"Building taxonomy neighbors (KNN mode) for {self.n_bins} bins with K={K}...")
        
        # Encode taxonomy levels as integers for fast comparison
        tax_groups = {}  # level -> {value -> list of bin indices}
        tax_codes = {}   # level -> array of codes for each bin
        
        for level in self.tax_levels:
            if level not in self.bins.columns:
                tax_codes[level] = np.full(self.n_bins, -1, dtype=np.int32)
                tax_groups[level] = {}
                continue
            
            codes, uniques = pd.factorize(self.bins[level], sort=False)
            tax_codes[level] = codes
            
            groups = {}
            for idx, code in enumerate(codes):
                if code >= 0:
                    if code not in groups:
                        groups[code] = []
                    groups[code].append(idx)
            tax_groups[level] = groups
        
        # For each bin, find K nearest neighbors by taxonomy
        for i in tqdm(range(self.n_bins), desc="Building neighbors (KNN)"):
            candidates = []  # list of (distance, bin_idx)
            seen = {i}  # exclude self
            
            for d, level in enumerate(self.tax_levels, start=1):
                code = tax_codes[level][i]
                if code < 0:
                    continue
                
                same_group = tax_groups[level].get(code, [])
                for j in same_group:
                    if j not in seen:
                        candidates.append((d, j))
                        seen.add(j)
            
            # If still not enough, add remaining bins with max distance
            if len(candidates) < K:
                for j in range(self.n_bins):
                    if j not in seen:
                        candidates.append((len(self.tax_levels) + 1, j))
                        if len(candidates) >= K:
                            break
            
            # Sort by distance and take top K
            candidates.sort(key=lambda x: x[0])
            top_k = candidates[:K]
            
            self.neighbours[i] = [x[1] for x in top_k]
            self.distances[i] = np.array([x[0] for x in top_k], dtype=float)
        
        # Log statistics
        neighbor_counts = [len(n) for n in self.neighbours]
        log.debug(
            f"Neighbor count stats: min={min(neighbor_counts)}, "
            f"max={max(neighbor_counts)}, mean={np.mean(neighbor_counts):.1f}")

    def build_taxonomy_neighbors_threshold(self, dist_threshold: int) -> None:
        """
        Populate neighbours using taxonomic discrete distance with a threshold.
        Selects all neighbors within the given taxonomic distance threshold.
        
        Args:
            dist_threshold: Maximum taxonomic distance to include as neighbor.
                Distance 1 = same species, 2 = same genus, etc.
        """
        from tqdm import tqdm
        if "bin_uri" not in self.bins.columns:
            raise ValueError("bins dataframe must contain 'bin_uri'")
        
        log.debug(f"Building taxonomy neighbors for {self.n_bins} bins with threshold {dist_threshold}...")
        
        # Encode taxonomy levels as integers for fast comparison
        # Build lookup tables for each level: bins grouped by their taxonomy value
        tax_groups = {}  # level -> {value -> list of bin indices}
        tax_codes = {}   # level -> array of codes for each bin
        
        for level in self.tax_levels:
            if level not in self.bins.columns:
                # Use -1 as missing-taxonomy sentinel, matching pandas.factorize NaN behavior.
                tax_codes[level] = np.full(self.n_bins, -1, dtype=np.int32)
                tax_groups[level] = {}
                continue
            
            # Factorize the taxonomy column (converts to integer codes, NaN -> -1)
            codes, uniques = pd.factorize(self.bins[level], sort=False)
            tax_codes[level] = codes  # np array of int codes, same length as n_bins
            
            # Build reverse lookup: code -> list of bin indices
            groups = {}
            for idx, code in enumerate(codes):
                if code >= 0:  # skip NaN values
                    if code not in groups:
                        groups[code] = []
                    groups[code].append(idx)
            tax_groups[level] = groups  # {tax_code: [list of bin indices with this tax_code]}
        
        # For each bin, find all neighbors within the distance threshold
        # Strategy: iterate through taxonomy levels up to dist_threshold,
        # collecting all bins that share taxonomy at each level
        
        for i in tqdm(range(self.n_bins), desc="Building neighbors"):
            candidates = []  # list of (distance, bin_idx)
            seen = {i}  # exclude self
            
            # Only iterate through levels up to dist_threshold
            for d, level in enumerate(self.tax_levels[:dist_threshold], start=1):
                code = tax_codes[level][i]
                if code < 0:  # NaN at this level
                    continue
                
                # Get all bins with same taxonomy value at this level
                same_group = tax_groups[level].get(code, [])
                for j in same_group:
                    if j not in seen:
                        candidates.append((d, j))
                        seen.add(j)
            
            # Sort by distance (closest first)
            candidates.sort(key=lambda x: x[0])
            
            self.neighbours[i] = [x[1] for x in candidates]
            self.distances[i] = np.array([x[0] for x in candidates], dtype=float)
        
        # Log statistics about neighbor counts
        neighbor_counts = [len(n) for n in self.neighbours]
        log.debug(
            f"Neighbor count stats: min={min(neighbor_counts)}, "
            f"max={max(neighbor_counts)}, mean={np.mean(neighbor_counts):.1f}, "
            f"median={np.median(neighbor_counts):.1f}"
        )

    def analyze_taxonomy_thresholds(self) -> pd.DataFrame:
        """
        Analyze neighbor counts for each possible taxonomic distance threshold.
        
        Returns a DataFrame with statistics (min, max, mean, std) for each threshold,
        helping you choose an appropriate dist_thres value.
        
        Returns:
            pd.DataFrame with columns: threshold, level, min, max, mean, std, median, pct_zero (percentage of bins with 0 neighbors)
        """
        from tqdm import tqdm
        
        if "bin_uri" not in self.bins.columns:
            raise ValueError("bins dataframe must contain 'bin_uri'")
        
        # Encode taxonomy levels
        tax_groups = {}
        tax_codes = {}
        
        for level in self.tax_levels:
            if level not in self.bins.columns:
                tax_codes[level] = np.full(self.n_bins, -1, dtype=np.int32)
                tax_groups[level] = {}
                continue
            
            codes, uniques = pd.factorize(self.bins[level], sort=False)
            tax_codes[level] = codes
            
            groups = {}
            for idx, code in enumerate(codes):
                if code >= 0:
                    if code not in groups:
                        groups[code] = []
                    groups[code].append(idx)
            tax_groups[level] = groups
        
        # Compute cumulative neighbor counts for each threshold
        results = []
        
        for threshold in range(1, len(self.tax_levels) + 1):
            level_name = self.tax_levels[threshold - 1]
            neighbor_counts = []
            
            for i in tqdm(range(self.n_bins), desc=f"Analyzing threshold {threshold} ({level_name})", leave=False):
                seen = {i}
                count = 0
                
                for d, level in enumerate(self.tax_levels[:threshold], start=1):
                    code = tax_codes[level][i]
                    if code < 0:
                        continue
                    
                    same_group = tax_groups[level].get(code, [])
                    for j in same_group:
                        if j not in seen:
                            count += 1
                            seen.add(j)
                
                neighbor_counts.append(count)
            
            counts_arr = np.array(neighbor_counts)
            
            # Calculate percentage of missing values at this level
            has_taxonomy = tax_codes[level_name] >= 0
            pct_missing = round(100 * (~has_taxonomy).sum() / self.n_bins, 2)
            
            # Min excluding bins with missing taxonomy at this level
            counts_with_tax = counts_arr[has_taxonomy]
            min_val = int(counts_with_tax.min()) if len(counts_with_tax) > 0 else None
            
            results.append({
                'threshold': threshold,
                'level': level_name,
                'pct_missing': pct_missing,
                'min': min_val,
                'max': int(counts_arr.max()),
                'mean': round(counts_arr.mean(), 2),
                'std': round(counts_arr.std(), 2),
                'median': round(np.median(counts_arr), 1),
                'pct_zero': round(100 * (counts_arr == 0).sum() / len(counts_arr), 2)
            })
        
        df = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("TAXONOMY DISTANCE THRESHOLD ANALYSIS")
        print("=" * 80)
        print(f"\nTotal bins: {self.n_bins}")
        print(f"Taxonomy levels: {self.tax_levels}\n")
        print(df.to_string(index=False))
        print("\n" + "=" * 80)
        print("Interpretation:")
        print("  - threshold 1 (species): neighbors share the same species")
        print("  - threshold 2 (genus): neighbors share species OR genus")
        print("  - threshold 3 (subfamily): neighbors share species OR genus OR subfamily")
        print("  - etc.")
        print("  - pct_missing: percentage of bins with missing taxonomy at this level")
        print("  - pct_zero: percentage of bins with NO neighbors at this threshold")
        print("=" * 80 + "\n")
        
        return df

    def _build_taxonomy_neighbors_for_subset(self, subset_indices: np.ndarray, K: int) -> None:
        """
        Compute KNN taxonomy neighbors for a specific subset of bin indices and write
        the results into self.neighbours / self.distances only for those indices.
        Used as a fallback for bins that lack DNA sequences.

        Args:
            subset_indices: Array of global bin indices that need taxonomy neighbors.
            K: Number of nearest neighbors to select.
        """
        from tqdm import tqdm

        # Build (or reuse) taxonomy code maps
        tax_codes: dict = {}
        tax_groups: dict = {}
        for level in self.tax_levels:
            if level not in self.bins.columns:
                tax_codes[level] = np.full(self.n_bins, -1, dtype=np.int32)
                tax_groups[level] = {}
                continue
            codes, _ = pd.factorize(self.bins[level], sort=False)
            tax_codes[level] = codes
            groups: dict = {}
            for idx, code in enumerate(codes):
                if code >= 0:
                    groups.setdefault(code, []).append(idx)
            tax_groups[level] = groups

        for i in tqdm(subset_indices, desc="Taxonomy fallback for bins without sequences", leave=False):
            candidates = []
            seen = {int(i)}
            for d, level in enumerate(self.tax_levels, start=1):
                code = tax_codes[level][i]
                if code < 0:
                    continue
                for j in tax_groups[level].get(code, []):
                    if j not in seen:
                        candidates.append((d, j))
                        seen.add(j)
            # If still not enough, pad with remaining bins
            if len(candidates) < K:
                for j in range(self.n_bins):
                    if j not in seen:
                        candidates.append((len(self.tax_levels) + 1, j))
                        seen.add(j)
                        if len(candidates) >= K:
                            break
            candidates.sort(key=lambda x: x[0])
            top_k = candidates[:K]
            self.neighbours[i] = [x[1] for x in top_k]
            self.distances[i] = np.array([float(x[0]) for x in top_k], dtype=float)

    def build_embedding_neighbors_knn(self, K: int) -> None:
        """
        Build KNN based on embeddings using sklearn.NearestNeighbors.

        Cosine distance is computed by L2-normalizing embeddings before Euclidean
        nearest-neighbor search.
        Bins without embeddings fall back to taxonomy-based KNN.

        Note:
        - For bins with embeddings, exactly min(K, n_with_embeddings - 1) neighbors are
            selected after removing self from the query result.
        - For bins without embeddings, neighbors are filled by taxonomy fallback.

        Args:
            K: Number of nearest neighbors to select.
        """
        if self.embeddings is None:
            raise ValueError(
                "embeddings not available — config.use_embedding=True but no embeddings were "
                "provided to NeighbourGraph.__init__()"
            )

        emb_indices = np.where(self.bins_with_embedding)[0]
        fallback_indices = np.where(~self.bins_with_embedding)[0]

        if len(emb_indices) == 0:
            raise ValueError(
                "No bins have embeddings. Check that bin_uri values in cfg.embedding_path "
                "match those in the training data."
            )

        # Optionally L2-normalise for cosine distance
        emb = self.embeddings.copy()
        if getattr(self.cfg, "emb_distance_metric", "cosine") == "cosine":
            emb = normalize(emb, norm="l2")

        # Fit NearestNeighbors on the subset that actually has sequences
        emb_subset = emb[emb_indices]  # [n_with_emb, dim]
        n_neighbors_query = min(K + 1, len(emb_indices))  # +1 to exclude self in the result
        nbrs = NearestNeighbors(n_neighbors=n_neighbors_query, algorithm="auto", metric="euclidean")
        nbrs.fit(emb_subset)

        # Query from the same subset (no cross-queries; fallback bins handled separately)
        distances, local_indices = nbrs.kneighbors(emb_subset)

        for rank, global_i in enumerate(emb_indices):
            # local_indices[rank, 0] == rank (self) → skip first column
            neighbor_local = local_indices[rank, 1:]
            neighbor_global = emb_indices[neighbor_local]
            self.neighbours[global_i] = neighbor_global.tolist()
            self.distances[global_i] = distances[rank, 1:]

        log.debug(
            f"Built embedding neighbors (KNN, K={K}, metric={self.cfg.emb_distance_metric}) "
            f"for {len(emb_indices)} bins."
        )

        # Taxonomy fallback for bins with no sequence
        if len(fallback_indices) > 0:
            log.debug(f"Running taxonomy fallback for {len(fallback_indices)} bins without sequences.")
            self._build_taxonomy_neighbors_for_subset(fallback_indices, K=K)

    def build_embedding_neighbors_threshold(self, radius: float) -> None:
        """
        Build radius-based neighbors using embeddings (BallTree).

        Cosine distance is computed by L2-normalizing embeddings before a Euclidean
        radius query (d_euclidean on unit sphere = sqrt(2 - 2*cos) ≈ cosine distance).
        Bins without embeddings fall back to taxonomy-based threshold neighbors.

        Args:
            radius: Maximum embedding distance to include as neighbor.
        """
        if self.embeddings is None:
            raise ValueError(
                "embeddings not available — config.use_embedding=True but no embeddings were "
                "provided to NeighbourGraph.__init__()"
            )

        emb_indices = np.where(self.bins_with_embedding)[0]
        fallback_indices = np.where(~self.bins_with_embedding)[0]

        if len(emb_indices) == 0:
            raise ValueError(
                "No bins have embeddings. Check that bin_uri values in cfg.embedding_path "
                "match those in the training data."
            )

        # Optionally L2-normalise for cosine distance
        emb = self.embeddings.copy()
        if getattr(self.cfg, "emb_distance_metric", "cosine") == "cosine":
            emb = normalize(emb, norm="l2")

        emb_subset = emb[emb_indices]
        tree = BallTree(emb_subset, metric="euclidean")
        local_indices_arr, distances_arr = tree.query_radius(
            emb_subset, r=radius, return_distance=True, sort_results=True
        )

        for rank, global_i in enumerate(emb_indices):
            local_nbrs = local_indices_arr[rank]
            dists = distances_arr[rank]
            # Remove self (distance 0)
            not_self = local_nbrs != rank
            self.neighbours[global_i] = emb_indices[local_nbrs[not_self]].tolist()
            self.distances[global_i] = dists[not_self]

        neighbor_counts = [len(self.neighbours[i]) for i in emb_indices]
        log.debug(
            f"Embedding threshold neighbors (radius={radius}, metric={self.cfg.emb_distance_metric}): "
            f"min={min(neighbor_counts)}, max={max(neighbor_counts)}, "
            f"mean={np.mean(neighbor_counts):.1f}, median={np.median(neighbor_counts):.1f}"
        )

        # Taxonomy fallback for bins with no sequence
        if len(fallback_indices) > 0:
            log.debug(f"Running taxonomy fallback for {len(fallback_indices)} bins without sequences.")
            self._build_taxonomy_neighbors_for_subset(
                fallback_indices, K=self.cfg.K  # use K as a reasonable default for fallback
            )

    def build_hybrid_neighbors(self, tax_threshold: int, emb_radius: float, K: int) -> None:
        """Build neighbors using both taxonomy and embedding distances."""
        raise NotImplementedError("Hybrid neighbor graph not yet implemented")

    def build(self) -> None:
        """
        Build the neighbor graph based on cfg.neighbor_mode.
        
        Modes:
        - "threshold": uses cfg.dist_thres (taxonomy) or cfg.emb_radius (embedding)
        - "knn": uses cfg.K for K-nearest neighbors
        """
        mode = getattr(self.cfg, 'neighbor_mode', 'threshold')
        
        if self.cfg.use_taxonomy and not self.cfg.use_embedding:
            if mode == "knn":
                self.build_taxonomy_neighbors_knn(self.cfg.K)
            else:
                self.build_taxonomy_neighbors_threshold(self.cfg.dist_thres)
        elif self.cfg.use_embedding and not self.cfg.use_taxonomy:
            if mode == "knn":
                self.build_embedding_neighbors_knn(self.cfg.K)
            else:
                self.build_embedding_neighbors_threshold(self.cfg.emb_radius)
        else:
            # Hybrid mode
            self.build_hybrid_neighbors(self.cfg.dist_thres, self.cfg.emb_radius, self.cfg.K)

    # --------- kernel weights (Nadaraya-Watson)
    def compute_kernel_q(self) -> float:
        """Compute an adaptive q if not provided: q = 1 / median(dist_to_Kth_neighbor^2)
        Works for continuous distances; for discrete taxonomic distances will return 1.
        """
        if self.cfg.kernel_q is not None:
            return self.cfg.kernel_q
        # use median of last neighbour distance
        last_dists = np.array([d[-1] if len(d) > 0 else 1.0 for d in self.distances])
        med = np.median(last_dists)
        if med <= 0:
            return 1.0
        # Floor med to prevent q = 1/med² from overflowing exp(-q·dist²) when
        # distances are very small (e.g. near-identical sequences).
        med = max(med, 1e-6)
        return 1.0 / (med ** 2)

    def nw_weights_for_node(self, i: int, q: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return neighbor indices and kernel-normalized weights for node i.
        Returns (indices_array, weights_array).
        """
        idx = np.array(self.neighbours[i])
        if len(idx) == 0:
            return idx, np.array([])
        dists = np.array(self.distances[i], dtype=float)
        if q is None:
            q = self.compute_kernel_q()
        w_unnorm = np.exp(-q * (dists ** 2))
        w = w_unnorm / (w_unnorm.sum() + 1e-12)
        return idx, w

    # -------- LLR coefficients per node
    def llr_coeffs_for_node(self, i: int, q: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return local-linear interpolation coefficients for node i.

        Returns (indices, coeffs) such that h_i ≈ sum_j coeffs[j] * h_j over neighbors.
        Coeffs are derived from weighted least squares on local offsets X_j - X_i and
        correspond to the intercept row e0^T (Z^T W Z)^(-1) Z^T W.

        If the local normal matrix is ill-conditioned, this falls back to NW weights.
        """
        idx = np.array(self.neighbours[i])
        if len(idx) == 0 or self.embeddings is None:
            return idx, np.array([])
        X = self.embeddings[idx] - self.embeddings[i]
        # build design Z = [1 | X]
        ones = np.ones((len(idx), 1))
        Z = np.hstack([ones, X])
        if q is None:
            q = self.compute_kernel_q()
        dists = np.linalg.norm(self.embeddings[idx] - self.embeddings[i], axis=1)
        w = np.exp(-q * (dists ** 2))
        W = np.diag(w)
        # solve for beta = (Z^T W Z)^{-1} Z^T W d  ; but we only need intercept effect on d_j
        # The interpolation matrix row that maps d -> a_i is: coeffs = e_0^T (Z^T W Z)^{-1} Z^T W
        # We'll compute coeffs row explicitly by solving (Z^T W Z)^T x = e_0 -> x obtains coefficients
        A = Z.T @ W @ Z
        try:
            invA = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            # ill-conditioned -> fallback to NW weights
            idx_nw, w_nw = self.nw_weights_for_node(i, q)
            return idx_nw, w_nw
        # row vector: e0^T invA Z^T W  (shape 1 x n_idx)
        e0 = np.zeros((A.shape[0],))
        e0[0] = 1.0
        row = e0 @ invA @ Z.T @ W
        return idx, row

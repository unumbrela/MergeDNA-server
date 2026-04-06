"""Merge pattern analysis and interpretability tools.

Analyses the source matrix S produced by MergeDNA's Local Encoder to
understand what the model has learned about genomic structure:

1. Extract merge boundary positions from the source matrix
2. Compute overlap with biological annotations (ENCODE, gene boundaries)
3. Visualise merge patterns across different genomic contexts
4. Map SAE features to biological function categories
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class MergePatternAnalyzer:
    """Analyse merge patterns produced by MergeDNA against biological annotations.

    The source matrix S[b, i, j] = 1 means original position i was merged
    into token j.  Merge boundaries occur where consecutive positions map
    to different merged tokens — these are the learned "word boundaries"
    in DNA.
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_merge_boundaries(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Extract merge boundary positions from the source matrix.

        A boundary occurs at position i if position i and i+1 map to
        different merged tokens (i.e., they were NOT merged together).

        Args:
            input_ids: [B, N] token IDs.
            attention_mask: [B, N] optional mask.

        Returns:
            List of lists: boundary positions for each sample in batch.
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        out = self.model.forward_with_intermediates(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        source = out["source"]  # [B, N, L]

        boundaries = []
        B, N, L = source.shape
        for b in range(B):
            # For each position, find which merged token it belongs to
            # (the one with highest weight in the source row)
            token_assignment = source[b].argmax(dim=-1)  # [N]
            # Boundary = where assignment changes
            boundary_pos = []
            for i in range(N - 1):
                if token_assignment[i] != token_assignment[i + 1]:
                    boundary_pos.append(i + 1)  # boundary after position i
            boundaries.append(boundary_pos)

        return boundaries

    @torch.no_grad()
    def compute_merge_stats(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute statistics about merging behaviour.

        Returns:
            Dict with:
                avg_token_length: average number of bases per merged token
                compression_ratio: L / N
                num_tokens: number of merged tokens
                boundary_count: number of merge boundaries
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        out = self.model.forward_with_intermediates(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        source = out["source"]  # [B, N, L]
        B, N, L = source.shape

        # Group sizes: how many original bases map to each merged token
        group_sizes = source.sum(dim=1)  # [B, L] — bases per token

        stats = {
            "avg_token_length": group_sizes[group_sizes > 0].mean().item(),
            "compression_ratio": L / N,
            "num_tokens": L,
            "input_length": N,
        }

        boundaries = self.extract_merge_boundaries(input_ids, attention_mask)
        stats["avg_boundaries"] = np.mean([len(b) for b in boundaries])

        return stats

    def compute_token_length_distribution(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 16,
    ) -> Dict[int, float]:
        """Compute distribution of merged token lengths (like MergeDNA Fig. 3).

        Returns:
            Dict mapping token_length -> normalised frequency.
        """
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            out = self.model.forward_with_intermediates(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            source = out["source"]  # [B, N, L]

        group_sizes = source.sum(dim=1)  # [B, L]
        # Flatten and count
        sizes = group_sizes[group_sizes > 0].cpu().numpy().astype(int)

        counts = defaultdict(int)
        for s in sizes:
            k = min(s, max_length)
            counts[k] += 1

        total = sum(counts.values())
        dist = {k: v / total for k, v in sorted(counts.items())}
        return dist

    @staticmethod
    def compute_annotation_overlap(
        boundaries: List[int],
        annotations: List[Tuple[int, int]],
        tolerance: int = 5,
    ) -> Dict[str, float]:
        """Compute overlap between merge boundaries and biological annotations.

        Args:
            boundaries: List of boundary positions.
            annotations: List of (start, end) annotation regions.
            tolerance: Allowed distance (bp) for a boundary to "match" an
                annotation edge.

        Returns:
            Dict with precision, recall, f1 of boundary-annotation alignment.
        """
        if not boundaries or not annotations:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Annotation edges
        annot_edges = set()
        for start, end in annotations:
            annot_edges.add(start)
            annot_edges.add(end)

        boundary_set = set(boundaries)

        # A boundary is a true positive if it's within `tolerance` of an edge
        tp_boundary = 0
        for b in boundary_set:
            for edge in annot_edges:
                if abs(b - edge) <= tolerance:
                    tp_boundary += 1
                    break

        tp_edge = 0
        for edge in annot_edges:
            for b in boundary_set:
                if abs(b - edge) <= tolerance:
                    tp_edge += 1
                    break

        precision = tp_boundary / max(len(boundary_set), 1)
        recall = tp_edge / max(len(annot_edges), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {"precision": precision, "recall": recall, "f1": f1}

    def compare_teacher_student_patterns(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compare merge patterns between teacher and student.

        Returns:
            Dict with boundary_overlap (IoU), assignment_cosine_sim.
        """
        # Get boundaries from both models
        self.model = teacher_model
        teacher_bounds = self.extract_merge_boundaries(input_ids, attention_mask)

        self.model = student_model
        student_bounds = self.extract_merge_boundaries(input_ids, attention_mask)

        # Compute IoU of boundary sets
        overlaps = []
        for t_b, s_b in zip(teacher_bounds, student_bounds):
            t_set = set(t_b)
            s_set = set(s_b)
            if not t_set and not s_set:
                overlaps.append(1.0)
            else:
                inter = len(t_set & s_set)
                union = len(t_set | s_set)
                overlaps.append(inter / max(union, 1))

        return {
            "boundary_iou": np.mean(overlaps),
        }

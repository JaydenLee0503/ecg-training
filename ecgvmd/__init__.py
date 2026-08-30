"""ecgvmd - VMD-based feature extraction for the ECGData.mat arrhythmia set.

The pipeline, in order:

    load_ecgdata()      ECGData.mat            -> 162 records x 65536 samples
    segment()           records                -> (B, 500) windows + labels + GROUPS
    extract_features()  windows                -> IMF feature blocks (FeatureBundle)
    evaluate()          features               -> record-wise cross-validated scores
    MRMRSelector        ~250 features          -> a qubit-sized subset

Typical use:

    from ecgvmd import load_ecgdata, segment, extract_features, evaluate, CFG
    ds = load_ecgdata()
    W, y, g = segment(ds, CFG)
    fb = extract_features(W, y, g, CFG)
    print(evaluate(None, *fb["VMD modes + rhythm"][:1], y, g))
"""
from .config import CFG, FAST, Config, CLASS_ORDER, CLASS_COLORS, EPS
from .data import ECGDataset, load_ecgdata, find_mat, stratified_record_sample
from .vmd import VMDResult, vmd, vmd_batch, vmd_apply
from .segment import (detect_r_peaks, heart_rate, fixed_windows, beat_windows,
                      segment, standardise)
from .features import (FeatureBundle, extract_features, mode_features, rr_features,
                       global_features, raw_features, perm_entropy, higuchi_fd,
                       MODE_FEATS, RR_NAMES, GLOBAL_NAMES)
from .evaluate import (evaluate, group_cv, naive_cv, record_vote, leakage_gap, rf,
                       default_models, compare_blocks, report)
from .select import rank_anova, mrmr_select, MRMRSelector, quantum_ready
from .quantum import (TanhAngleScaler, angle_kernel_qnode, iqp_kernel_qnode,
                      product_angle_kernel, gram_matrix, QuantumKernelSVC,
                      vqc_qnode, VQCClassifier)

__version__ = "0.1.0"

__all__ = [
    "CFG", "FAST", "Config", "CLASS_ORDER", "CLASS_COLORS", "EPS",
    "ECGDataset", "load_ecgdata", "find_mat", "stratified_record_sample",
    "VMDResult", "vmd", "vmd_batch", "vmd_apply",
    "detect_r_peaks", "heart_rate", "fixed_windows", "beat_windows", "segment",
    "standardise",
    "FeatureBundle", "extract_features", "mode_features", "rr_features",
    "global_features", "raw_features", "perm_entropy", "higuchi_fd",
    "MODE_FEATS", "RR_NAMES", "GLOBAL_NAMES",
    "evaluate", "group_cv", "naive_cv", "record_vote", "leakage_gap", "rf",
    "default_models", "compare_blocks", "report",
    "rank_anova", "mrmr_select", "MRMRSelector", "quantum_ready",
    "TanhAngleScaler", "angle_kernel_qnode", "iqp_kernel_qnode",
    "product_angle_kernel", "gram_matrix", "QuantumKernelSVC",
    "vqc_qnode", "VQCClassifier",
]

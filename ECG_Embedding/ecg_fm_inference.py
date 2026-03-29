# ── Imports ───────────────────────────────────────────────────────────────────

import os
from itertools import chain
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb                          # PhysioNet WFDB format reader for .hea/.dat/.atr files
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from huggingface_hub import hf_hub_download

# ecg_transform handles the signal preprocessing pipeline (resampling, normalization, etc.)
from ecg_transform.inp import ECGInput, ECGInputSchema
from ecg_transform.sample import ECGMetadata, ECGSample
from ecg_transform.t.base import ECGTransform
from ecg_transform.t.common import HandleConstantLeads, LinearResample, ReorderLeads
from ecg_transform.t.scale import Standardize
from ecg_transform.t.cut import SegmentNonoverlapping

# fairseq_signals is the framework ECG-FM is built on top of
from fairseq_signals.utils.checkpoint_utils import load_model_and_task


# ── Constants ─────────────────────────────────────────────────────────────────

# ECG-FM was pretrained with this specific lead ordering. Any input ECG must be
# reordered to match this exactly, otherwise leads map to the wrong channels.
ECG_FM_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# ECG-FM expects signals at 500 Hz. Input recordings at other rates (e.g. 250 Hz,
# 1000 Hz) will be resampled to this target during preprocessing.
SAMPLE_RATE = 500

# The model processes fixed-length windows. 5 seconds at 500 Hz = 2500 samples.
# Longer recordings are split into non-overlapping 5-second segments.
N_SAMPLES   = SAMPLE_RATE * 5   # 5 seconds


# ── WFDB Loader ───────────────────────────────────────────────────────────────

def load_wfdb_folder(folder: str) -> List[Tuple[str, np.ndarray, int, List[str]]]:
    """
    Scan a directory for WFDB records and load each one into memory.

    WFDB stores each record as a pair of files: a .hea header (metadata) and a
    .dat binary (signal). We use the .hea files as the index to discover records,
    then load the full record via wfdb.rdrecord.

    The signal is transposed from (samples, leads) → (leads, samples) to match
    the (channels, time) convention expected downstream.

    Returns a list of (record_name, signal, sample_rate, lead_names) tuples.
    Corrupt or unreadable records are skipped with a warning rather than crashing.
    """
    records = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".hea"):
            continue
        record_name = os.path.join(folder, fname[:-4])
        try:
            rec        = wfdb.rdrecord(record_name)
            signal     = rec.p_signal.T.astype(np.float32)  # → (leads, samples)
            lead_names = rec.sig_name
            records.append((record_name, signal, int(rec.fs), lead_names))
        except Exception as e:
            print(f"[WARN] Skipping {record_name}: {e}")
    return records


# ── Label Loader ──────────────────────────────────────────────────────────────

def load_label_definitions_from_records(
    records:        List[Tuple[str, np.ndarray, int, List[str]]],
    annotation_ext: str = 'atr',
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a label vocabulary by scanning all annotation files in the record set.

    WFDB annotations (.atr files) store beat-level labels as single character
    symbols (e.g. 'N' = normal, 'V' = PVC). We collect the union of all symbols
    across every record to define the full label space, then assign each a stable
    integer ID via a sorted DataFrame.

    Records without an annotation file are silently skipped — this is expected
    when a subset of recordings lacks ground truth labels.
    """
    all_labels = set()
    for record_name, _, _, _ in records:
        try:
            ann = wfdb.rdann(record_name, annotation_ext)
            all_labels.update(ann.symbol)
        except FileNotFoundError:
            # Not all records have annotations; that's fine
            pass
        except Exception as e:
            print(f"[WARN] Could not read annotation for {record_name}: {e}")

    # Sort for deterministic label_id assignment across runs
    label_names = sorted(all_labels)
    label_def   = pd.DataFrame(
        index = label_names,
        data  = {'label_id': range(len(label_names))}
    )
    label_def.index.name = 'name'
    return label_def, label_names


# ── Transforms ────────────────────────────────────────────────────────────────

def build_transforms(
    target_lead_order:  List[str] = None,
    target_sample_rate: int       = SAMPLE_RATE,
    n_samples:          int       = N_SAMPLES,
) -> List[ECGTransform]:
    """
    Define the preprocessing pipeline applied to every ECG before it enters the model.

    The order of transforms matters:
      1. ReorderLeads        — remap leads to the model's expected channel order first,
                               before any signal processing changes the data shape.
      2. HandleConstantLeads — zero out flat/disconnected leads so they don't produce
                               NaN after standardization (std = 0 → division by zero).
      3. LinearResample      — bring all recordings to the target sample rate so the
                               model sees a consistent time resolution.
      4. Standardize         — zero-mean, unit-variance per lead; removes amplitude
                               differences between devices and patients.
      5. SegmentNonoverlapping — chop the (now normalized) signal into fixed-length
                               windows that match the model's input size.
    """
    target_lead_order = target_lead_order or ECG_FM_LEAD_ORDER
    return [
        ReorderLeads(target_lead_order),
        HandleConstantLeads(),
        LinearResample(target_sample_rate),
        Standardize(),
        SegmentNonoverlapping(n_samples),
    ]


# ── Dataset ───────────────────────────────────────────────────────────────────

class ECGFMDataset(Dataset):
    """
    PyTorch Dataset that wraps pre-loaded WFDB records.

    Each record is one physical ECG file (which may be minutes long). __getitem__
    applies the full preprocessing pipeline and returns all non-overlapping segments
    for that record as a stacked tensor — so one record maps to one or more model
    inputs depending on its duration.

    We also return the raw ECGInput alongside the tensor so callers can trace each
    output segment back to its source file and position.
    """

    def __init__(
        self,
        schema:     ECGInputSchema,
        transforms: List[ECGTransform],
        records:    List[Tuple[str, np.ndarray, int, List[str]]],
    ):
        self.schema     = schema
        self.transforms = transforms
        self.records    = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ECGInput]:
        record_name, signal, fs, lead_names = self.records[idx]

        # ECGMetadata tells the transform pipeline what it's working with:
        # the original sample rate (needed for resampling), the lead layout
        # (needed for reordering), and the valid sample range.
        metadata = ECGMetadata(
            sample_rate = fs,
            num_samples = signal.shape[1],
            lead_names  = lead_names,
            unit        = None,
            input_start = 0,
            input_end   = signal.shape[1],
        )
        metadata.file = record_name  # preserved for traceability in downstream analysis

        # ECGInput pairs the raw signal array with its metadata.
        # ECGSample applies the full transform chain and exposes the result via .out.
        inp    = ECGInput(signal, metadata)
        sample = ECGSample(inp, self.schema, self.transforms)

        # sample.out is (n_segments, leads, samples) — convert to float32 tensor
        source = torch.from_numpy(sample.out).float()
        return source, inp


# ── Collation ─────────────────────────────────────────────────────────────────

def collate_fn(
    inps: List[Tuple[torch.Tensor, ECGInput]]
) -> Tuple[torch.Tensor, List[ECGInput]]:
    """
    Custom collate function to handle variable-length records in a batch.

    Each item from the Dataset is one record yielding N segments. Across items in
    a batch, N varies (longer recordings produce more segments). We therefore:
      - Concatenate all segment tensors along dim 0 into a flat
        (total_segments, leads, samples) batch.
      - Repeat the source ECGInput N times per record so every segment in the
        batch has a corresponding identifier pointing back to its origin file.

    This means the effective batch size seen by the model may differ from
    batch_size, which is set in terms of records rather than segments.
    """
    sample_ids = list(
        chain.from_iterable([[inp[1]] * inp[0].shape[0] for inp in inps])
    )
    return torch.concatenate([inp[0] for inp in inps]), sample_ids


# ── Loader Factory ────────────────────────────────────────────────────────────

def records_to_loader(
    records:     List[Tuple[str, np.ndarray, int, List[str]]],
    schema:      ECGInputSchema,
    transforms:  List[ECGTransform],
    batch_size:  int = 64,
    num_workers: int = 7,
) -> DataLoader:
    """
    Wrap a list of records in a DataLoader ready for inference.

    Key configuration choices:
      - shuffle=False    — preserves record order so embeddings line up predictably
                           with the input list for downstream analysis.
      - drop_last=False  — we want embeddings for every segment, even if the last
                           batch is smaller than batch_size.
      - pin_memory=True  — speeds up host→GPU transfers when using CUDA.
    """
    dataset = ECGFMDataset(schema=schema, transforms=transforms, records=records)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = True,
        sampler     = None,
        shuffle     = False,
        collate_fn  = collate_fn,
        drop_last   = False,
    )


# ── Model Loader ──────────────────────────────────────────────────────────────

def build_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load an ECG-FM model from a fairseq-signals checkpoint file.

    load_model_and_task can return either a single model or a list (e.g. when an
    ensemble was saved). We always take the first (or only) model. The model is
    moved to the target device and set to eval mode to disable dropout and
    BatchNorm training behaviour before inference.
    """
    models, cfg, task = load_model_and_task(checkpoint_path)

    # Normalise the return type — fairseq sometimes wraps a single model in a list
    model = models[0] if isinstance(models, list) else models

    model = model.to(device)
    model.eval()  # critical: disables dropout and sets BN to use running stats
    return model, cfg, task


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()  # disable gradient tracking — we only need forward pass values
def extract_embeddings(
    model:  torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, list]:
    """
    Run ECG-FM forward pass and collect embeddings + sample IDs.

    ECG-FM is a transformer encoder. Its forward pass returns a dict whose 'x'
    key holds the token-level hidden states with shape (B, T, C), where:
      B = batch size (segments)
      T = sequence length (number of time patches)
      C = hidden dimension (model width)

    We reduce the time axis by mean-pooling → (B, C), producing one fixed-size
    vector per segment. These vectors are the segment-level embeddings that can
    be fed into classifiers, clustering, or similarity search downstream.

    Returns:
        embeddings (np.ndarray) – shape (N_segments, hidden_dim)
        sample_ids (list)       – ECGInput per segment for traceability
    """
    all_embeddings = []
    all_sample_ids = []

    for batch_signals, batch_inps in loader:
        x          = batch_signals.to(device)
        net_output = model(source=x, padding_mask=None)

        # On the first batch, print output keys so the caller can verify the
        # model is producing the expected outputs (useful during development).
        if not all_embeddings:
            print(f"net_output keys: {list(net_output.keys())}")
            for k, v in net_output.items():
                shape = v.shape if isinstance(v, torch.Tensor) else type(v)
                print(f"  {k}: {shape}")

        # Mean-pool over the time dimension to get a single vector per segment.
        # This is a simple but effective aggregation strategy; alternatives include
        # CLS token extraction or attention-weighted pooling.
        embeddings = net_output["x"].mean(dim=1)   # (B, T, C) → (B, C)

        all_embeddings.append(embeddings.cpu().numpy())
        all_sample_ids.extend(batch_inps)

    # Stack all batches into one contiguous array for easy downstream use
    return np.concatenate(all_embeddings, axis=0), all_sample_ids


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Config ────────────────────────────────────────────────────────────────
    ECG_FOLDER      = "ecg"
    CHECKPOINT_PATH = "ckpts/mimic_iv_ecg_physionet_pretrained.pt"
    BATCH_SIZE      = 8
    NUM_WORKERS     = 0   # set to 0 for debugging; increase for production throughput
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    # Download the pretrained weights on first run. The PhysioNet-pretrained
    # checkpoint is preferred here because our data also comes from PhysioNet,
    # giving the best domain match without fine-tuning.
    if not os.path.exists(CHECKPOINT_PATH):
        print("Checkpoint not found — downloading from HuggingFace ...")
        os.makedirs("ckpts", exist_ok=True)
        hf_hub_download(
            repo_id   = "wanglab/ecg-fm",
            filename  = "mimic_iv_ecg_physionet_pretrained.pt",
            local_dir = "ckpts",
        )
        print(f"Downloaded to: {CHECKPOINT_PATH}")

    # ── Data ──────────────────────────────────────────────────────────────────
    # Limit to 5 records for a quick sanity check; remove the slice for a full run.
    records = load_wfdb_folder(ECG_FOLDER)[:5]
    print(f"\nLoaded {len(records)} records")
    for r in records:
        print(f"  {r[0]}  shape={r[1].shape}  fs={r[2]}  leads={r[3]}")

    # Build the label vocabulary from whatever annotations are present.
    # This is informational at inference time; required if training a classifier head.
    label_def, label_names = load_label_definitions_from_records(records, annotation_ext='atr')
    print(f"\nLabels ({len(label_names)}): {label_names}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    # The schema acts as a contract: it declares what the model expects so that
    # the transform pipeline can validate and reject incompatible inputs early.
    schema = ECGInputSchema(
        sample_rate         = SAMPLE_RATE,
        expected_lead_order = ECG_FM_LEAD_ORDER,
        min_num_samples     = N_SAMPLES,
        partial_leads       = True,   # allow recordings with fewer than 12 leads
    )
    transforms = build_transforms()
    loader     = records_to_loader(
        records     = records,
        schema      = schema,
        transforms  = transforms,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading model from {CHECKPOINT_PATH} ...")
    model, cfg, task = build_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)
    print(f"Model loaded : {type(model).__name__}")
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\nRunning inference ...")
    embeddings, sample_ids = extract_embeddings(model, loader, DEVICE)

    # embeddings[i] is the 1-D embedding vector for the i-th 5-second segment.
    # sample_ids[i] is the corresponding ECGInput, pointing back to the source file.
    print(f"\nEmbeddings shape : {embeddings.shape}")   # (N_segments, hidden_dim)
    print(f"Total segments   : {len(sample_ids)}")
    print(f"Embedding sample :\n{embeddings[0, :8]}")   # first 8 dims of first segment

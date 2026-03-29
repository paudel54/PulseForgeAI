# ── Imports ───────────────────────────────────────────────────────────────────

import os
import sys
import warnings
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# Resolve fairseq-signals from a local submodule
current_dir = Path(__file__).resolve().parent
src_path    = str(current_dir / 'fairseq-signals/fairseq-signals')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from ecg_transform.inp import ECGInput, ECGInputSchema
from ecg_transform.sample import ECGMetadata, ECGSample
from ecg_transform.t.base import ECGTransform
from ecg_transform.t.common import HandleConstantLeads, LinearResample, ReorderLeads
from ecg_transform.t.scale import Standardize
from ecg_transform.t.cut import SegmentNonoverlapping
from fairseq_signals.utils.checkpoint_utils import load_model_and_task

torch.backends.cudnn.benchmark = True


# ── Constants ─────────────────────────────────────────────────────────────────

ECG_FM_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SAMPLE_RATE       = 500
N_SAMPLES         = SAMPLE_RATE * 5
ECG_NATIVE_HZ     = 130
ACC_HZ            = 100
WINDOW_5S_ECG     = ECG_NATIVE_HZ * 5
WINDOW_30S_ECG    = ECG_NATIVE_HZ * 30
WINDOW_5S_ACC     = ACC_HZ * 5


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_wfdb_folder(folder: str) -> List[Tuple[str, np.ndarray, int, List[str]]]:
    """
    Scan a directory for WFDB records and load each one into memory.
    Returns (record_name, signal, sample_rate, lead_names) tuples.
    Signal is transposed to (leads, samples).
    """
    records = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".hea"):
            continue
        record_name = os.path.join(folder, fname[:-4])
        try:
            rec        = wfdb.rdrecord(record_name)
            signal     = rec.p_signal.T.astype(np.float32)
            lead_names = rec.sig_name
            records.append((record_name, signal, int(rec.fs), lead_names))
        except Exception as e:
            print(f"[WARN] Skipping {record_name}: {e}")
    return records


def load_acc_folder(folder: str) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Load ACC records and index by (patient_id, session).
    Returns dict keyed by ('001', '1') → (N_samples, 3) float32 array.
    """
    acc_map = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".hea") or "_acc" not in fname:
            continue
        record_name = os.path.join(folder, fname[:-4])
        try:
            rec    = wfdb.rdrecord(record_name)
            signal = rec.p_signal.astype(np.float32)
            parts  = fname[:-4].split('_')
            key    = (parts[0], parts[1])
            acc_map[key] = signal
        except Exception as e:
            print(f"[WARN] Skipping ACC {record_name}: {e}")
    return acc_map


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — ECG-FM PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_transforms(
    target_lead_order:  List[str] = None,
    target_sample_rate: int       = SAMPLE_RATE,
    n_samples:          int       = N_SAMPLES,
) -> List[ECGTransform]:
    """
    Preprocessing chain applied before the ECG-FM forward pass.
    Order is fixed — see ecg_fm_inference.py for full rationale.
    """
    target_lead_order = target_lead_order or ECG_FM_LEAD_ORDER
    return [
        ReorderLeads(target_lead_order),
        HandleConstantLeads(),
        LinearResample(target_sample_rate),
        Standardize(),
        SegmentNonoverlapping(n_samples),
    ]


class ECGFMDataset(Dataset):
    def __init__(self, schema, transforms, records):
        self.schema     = schema
        self.transforms = transforms
        self.records    = records

    def __len__(self):
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
    
        inp = ECGInput(signal, metadata)
    
        try:
            sample = ECGSample(inp, self.schema, self.transforms)
            source = torch.from_numpy(sample.out).float()
    
            # replace any NaN/inf that survive the transform chain with 0
            # rather than crashing the DataLoader worker process
            source = torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    
        except ValueError as e:
            # NaN values in signal — return a silent zero tensor of the expected
            # shape so collate_fn can still batch this item without crashing
            print(f"[WARN] Skipping segment in {record_name}: {e}")
            source = torch.zeros((1, len(ECG_FM_LEAD_ORDER), N_SAMPLES))
    
        return source, inp


def collate_fn(inps):
    sample_ids = list(chain.from_iterable([[inp[1]] * inp[0].shape[0] for inp in inps]))
    return torch.concatenate([inp[0] for inp in inps]), sample_ids


def records_to_loader(records, schema, transforms, batch_size=1, num_workers=4):
    dataset = ECGFMDataset(schema=schema, transforms=transforms, records=records)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = True,
        shuffle     = False,
        collate_fn  = collate_fn,
        drop_last   = False,
    )


def build_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load ECG-FM from a fairseq-signals checkpoint.
    torch.compile is omitted due to GCC 9.4 incompatibility on this cluster.
    """
    models, cfg, task = load_model_and_task(checkpoint_path)
    model = models[0] if isinstance(models, list) else models
    model = model.to(device)
    model.eval()
    return model, cfg, task


@torch.no_grad()
def extract_embeddings_chunked(
    model:      torch.nn.Module,
    segments:   torch.Tensor,
    device:     torch.device,
    chunk_size: int = 32,
) -> np.ndarray:
    """
    Run ECG-FM in chunk_size batches to avoid OOM on long recordings.

    The correct output key is 'features' — shape (B, T, 768) — which contains
    the transformer encoder representations. 'x' is the quantized codebook
    output used during pretraining and has an unstable last dimension.
    We mean-pool over the time dimension T → (B, 768).
    """
    all_embeddings = []

    for start in range(0, len(segments), chunk_size):
        chunk      = segments[start : start + chunk_size].to(device)
        net_output = model(source=chunk, padding_mask=None)

        # (B, T, 768) — mean-pool over time tokens → (B, 768)
        emb = net_output["features"].float().mean(dim=1)
        all_embeddings.append(emb.cpu().numpy())
        torch.cuda.empty_cache()

    if len(all_embeddings) == 0:
        return np.empty((0,), dtype=np.float32)

    return np.concatenate(all_embeddings, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — SIGNAL PROCESSING METRICS
#  Adapted from processing_worker.py for batch (non-streaming) use.
# ══════════════════════════════════════════════════════════════════════════════

def compute_acc_har_features(acc_xyz: np.ndarray, fs: float = ACC_HZ) -> dict:
    """
    Extract 4 HAR features from a 5-second ACC window.
    Features: mean_mag_mg, var_mag_mg2, spectral_entropy, median_freq_hz.
    """
    if acc_xyz is None or acc_xyz.ndim != 2 or acc_xyz.shape[1] < 3 or len(acc_xyz) < 20:
        return {"mean_mag_mg": None, "var_mag_mg2": None,
                "spectral_entropy": None, "median_freq_hz": None}

    mag      = np.sqrt(np.sum(acc_xyz[:, :3].astype(np.float64) ** 2, axis=1))
    N        = len(mag)
    mean_mag = float(np.mean(mag))
    var_mag  = float(np.var(mag, ddof=1))

    windowed = (mag - mean_mag) * np.hanning(N)
    fft_mag  = np.abs(np.fft.rfft(windowed))
    freqs    = np.fft.rfftfreq(N, d=1.0 / fs)
    cutoff   = np.searchsorted(freqs, 25.0)
    psd      = fft_mag[:cutoff] ** 2
    freqs    = freqs[:cutoff]
    total_p  = float(np.sum(psd))

    spectral_entropy = median_freq = None
    if total_p > 0 and len(freqs) > 1:
        pn               = psd / total_p
        spectral_entropy = float(-np.sum(pn * np.log2(pn + 1e-12)) / np.log2(len(pn)))
        cum_p            = np.cumsum(pn)
        med_idx          = int(np.searchsorted(cum_p, 0.5))
        median_freq      = float(freqs[min(med_idx, len(freqs) - 1)])

    def _r(v, d=4):
        return round(float(v), d) if v is not None and np.isfinite(v) else None

    return {
        "mean_mag_mg":      _r(mean_mag),
        "var_mag_mg2":      _r(var_mag),
        "spectral_entropy": _r(spectral_entropy),
        "median_freq_hz":   _r(median_freq, 3),
    }


def compute_5s_ecg_metrics(ecg_window: np.ndarray) -> dict:
    """
    Compute ECG SQI and instantaneous HR from a 5-second ECG window.
    Mirrors ProcessingWorker._compute_5s_window (CPU path only).
    """
    empty = {"sqi": None, "nk_sqi": None, "qrs_energy": None,
             "vital_kurtosis": None, "instant_hr": None, "n_r_peaks": 0}
    try:
        import neurokit2 as nk
        import scipy.signal

        b, a         = scipy.signal.butter(4, [0.5, 40], btype='bandpass', fs=ECG_NATIVE_HZ)
        ecg_f        = scipy.signal.filtfilt(b, a, ecg_window)
        ecg_c        = nk.ecg_clean(ecg_f, sampling_rate=ECG_NATIVE_HZ)
        _, peak_info = nk.ecg_peaks(ecg_c, sampling_rate=ECG_NATIVE_HZ)
        r_peaks      = np.array(peak_info.get("ECG_R_Peaks", []), dtype=int)

        if len(r_peaks) < 2:
            return empty

        qrs_energy = 0.5
        try:
            f, Pxx     = scipy.signal.welch(ecg_c, fs=ECG_NATIVE_HZ,
                                            nperseg=min(256, len(ecg_c)))
            qrs_p      = np.sum(Pxx[(f >= 5) & (f <= 15)])
            total_p    = np.sum(Pxx[(f >= 1) & (f <= 40)])
            if total_p > 0:
                qrs_energy = float(qrs_p / total_p)
        except Exception:
            pass

        nk_sqi = qrs_energy
        try:
            nk_sqi = float(np.mean(nk.ecg_quality(ecg_c, rpeaks=r_peaks,
                                                    sampling_rate=ECG_NATIVE_HZ)))
        except Exception:
            pass

        vital_kurtosis = None
        try:
            import vital_sqi.sqi.standard_sqi as standard_sqi
            if hasattr(standard_sqi, 'kurtosis_sqi'):
                k = standard_sqi.kurtosis_sqi(ecg_c)
                if k is not None:
                    vital_kurtosis = float(k)
        except Exception:
            pass

        rr_ms      = np.diff(r_peaks) / ECG_NATIVE_HZ * 1000.0
        valid_rr   = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
        instant_hr = float(60_000.0 / np.mean(valid_rr)) if len(valid_rr) > 0 else None

        return {
            "sqi":            round(nk_sqi, 4),
            "nk_sqi":         round(nk_sqi, 4),
            "qrs_energy":     round(qrs_energy, 4),
            "vital_kurtosis": round(vital_kurtosis, 2) if vital_kurtosis is not None else None,
            "instant_hr":     round(instant_hr, 1)     if instant_hr is not None else None,
            "n_r_peaks":      int(len(r_peaks)),
        }
    except Exception as e:
        return {**empty, "error_5s": str(e)}


def compute_30s_hrv_metrics(ecg_30s: np.ndarray) -> dict:
    """
    Compute full HRV metrics from a 30-second ECG window.
    Returns RMSSD, SDNN, mean_hr, LF/HF, and ECG morphology widths.
    Mirrors ProcessingWorker._compute_hrv.
    """
    empty = {"rmssd": None, "sdnn": None, "mean_hr": None, "lf_hf": None,
             "p_width": None, "qrs_width": None, "st_width": None,
             "qt_width": None, "qtc_width": None,
             "n_peaks_30s": 0, "hrv_status": "error"}
    try:
        import neurokit2 as nk
        import scipy.signal

        b, a         = scipy.signal.butter(4, [0.5, 40], btype='bandpass', fs=ECG_NATIVE_HZ)
        ecg_f        = scipy.signal.filtfilt(b, a, ecg_30s)
        ecg_c        = nk.ecg_clean(ecg_f, sampling_rate=ECG_NATIVE_HZ)
        _, peak_info = nk.ecg_peaks(ecg_c, sampling_rate=ECG_NATIVE_HZ)
        r_peaks      = np.array(peak_info.get("ECG_R_Peaks", []), dtype=int)

        if len(r_peaks) < 4:
            return {**empty, "n_peaks_30s": len(r_peaks), "hrv_status": "insufficient_peaks"}

        rr    = np.diff(r_peaks) / ECG_NATIVE_HZ * 1000.0
        valid = rr[(rr > 300) & (rr < 2000)]

        if len(valid) < 3:
            return {**empty, "n_peaks_30s": len(r_peaks), "hrv_status": "insufficient_rr"}

        rmssd   = float(np.sqrt(np.mean(np.diff(valid) ** 2)))
        sdnn    = float(np.std(valid, ddof=1))
        mean_rr = float(np.mean(valid))
        mean_hr = 60_000.0 / mean_rr if mean_rr > 0 else None

        lf_hf = None
        if len(valid) >= 10:
            try:
                import pyhrv.frequency_domain as fd
                res = fd.lomb_psd(nni=valid.tolist(), show=False,
                                  show_param=False, legend=False)
                v = res.get("lomb_ratio")
                if v is not None and np.isfinite(float(v)):
                    lf_hf = float(v)
            except Exception:
                pass

        p_width = qrs_width = st_width = qt_width = qtc_width = None
        try:
            _, waves = nk.ecg_delineate(ecg_c, r_peaks, sampling_rate=ECG_NATIVE_HZ,
                                         method="dwt", show=False)

            def _mean_width(on_key, off_key):
                ons  = waves.get(on_key, [])
                offs = waves.get(off_key, [])
                if not ons or not offs or len(ons) != len(offs):
                    return None
                ws = [off - on for on, off in zip(ons, offs)
                      if on is not None and off is not None
                      and not np.isnan(on) and not np.isnan(off)]
                return float(np.mean(ws)) / ECG_NATIVE_HZ * 1000 if ws else None

            p_width   = _mean_width("ECG_P_Onsets",  "ECG_P_Offsets")
            qrs_width = _mean_width("ECG_R_Onsets",  "ECG_R_Offsets")
            st_width  = _mean_width("ECG_R_Offsets", "ECG_T_Onsets")
            qt_width  = _mean_width("ECG_R_Onsets",  "ECG_T_Offsets")
            if qt_width is not None and mean_rr > 0:
                qtc_width = qt_width / np.sqrt(mean_rr / 1000.0)
        except Exception:
            pass

        def _r(v, d=2):
            return round(v, d) if v is not None and np.isfinite(v) else None

        return {
            "rmssd":       _r(rmssd),
            "sdnn":        _r(sdnn),
            "mean_hr":     _r(mean_hr, 1),
            "lf_hf":       _r(lf_hf, 3),
            "p_width":     _r(p_width, 1),
            "qrs_width":   _r(qrs_width, 1),
            "st_width":    _r(st_width, 1),
            "qt_width":    _r(qt_width, 1),
            "qtc_width":   _r(qtc_width, 1),
            "n_peaks_30s": int(len(r_peaks)),
            "hrv_status":  "ok",
        }
    except Exception as e:
        return {**empty, "error_hrv": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PER-RECORD PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_record(
    record:      Tuple[str, np.ndarray, int, List[str]],
    acc_map:     Dict,
    model:       torch.nn.Module,
    schema:      ECGInputSchema,
    transforms:  List[ECGTransform],
    device:      torch.device,
    chunk_size:  int = 32,
    num_workers: int = 4,
) -> Optional[pd.DataFrame]:
    """
    Process a single WFDB record end-to-end and return a joined DataFrame.

    For each 5-second segment the output row contains:
      Identifiers   : record_name, patient_id, session, segment_index,
                      segment_start_s, segment_end_s, hrv_window_index
      Signal info   : sample_rate, lead_names, n_leads_active
      Annotation    : exercise_label (from .atr)
      ECG-FM        : emb_0 … emb_N
      5s ECG        : ecg_sqi, ecg_nk_sqi, ecg_qrs_energy, ecg_vital_kurtosis,
                      ecg_instant_hr, ecg_n_r_peaks
      5s ACC        : acc_mean_mag_mg, acc_var_mag_mg2, acc_spectral_entropy,
                      acc_median_freq_hz
      30s HRV       : hrv_rmssd, hrv_sdnn, hrv_mean_hr, hrv_lf_hf,
                      hrv_p_width, hrv_qrs_width, hrv_st_width,
                      hrv_qt_width, hrv_qtc_width, hrv_n_peaks_30s,
                      hrv_status

    Returns None if the record produces no segments after preprocessing.
    """
    record_name, signal, fs, lead_names = record

    stem       = os.path.basename(record_name)
    parts      = stem.split('_')
    patient_id = parts[0]
    session    = parts[1] if len(parts) > 1 else '1'

    # ── GPU: ECG-FM embeddings ─────────────────────────────────────────────
    loader = records_to_loader(
        records     = [record],
        schema      = schema,
        transforms  = transforms,
        batch_size  = 1,
        num_workers = num_workers,
    )
    segments_tensor, sample_ids = next(iter(loader))
    n_segments = len(segments_tensor)

    embeddings = extract_embeddings_chunked(model, segments_tensor, device, chunk_size)

    if len(embeddings) == 0:
        return None

    # ── CPU: signal metrics ────────────────────────────────────────────────
    ecg_1d           = signal[0]
    n_native_samples = ecg_1d.shape[0]
    acc_signal       = acc_map.get((patient_id, session))

    # Load exercise annotations
    ann_events = []
    try:
        ann        = wfdb.rdann(record_name, 'atr')
        ann_events = list(zip(ann.sample, ann.aux_note if ann.aux_note else ann.symbol))
    except Exception:
        pass

    def _label_for_segment(start_s: float) -> str:
        start_sample = int(start_s * fs)
        label        = "rest"
        for sample, note in ann_events:
            if sample <= start_sample:
                label = str(note).strip()
        return label

    # Pre-compute 30s HRV windows and cache by window index
    hrv_cache: Dict[int, dict] = {}
    for w in range(n_native_samples // WINDOW_30S_ECG):
        start_samp   = w * WINDOW_30S_ECG
        ecg_30s      = ecg_1d[start_samp : start_samp + WINDOW_30S_ECG].astype(np.float64)
        hrv_cache[w] = compute_30s_hrv_metrics(ecg_30s)

    # ── Assemble per-segment rows ──────────────────────────────────────────
    rows = []

    for seg_idx in range(n_segments):
        start_s = seg_idx * N_SAMPLES / SAMPLE_RATE
        end_s   = start_s + N_SAMPLES / SAMPLE_RATE

        # 5s ECG metrics
        ecg_start      = int(start_s * ECG_NATIVE_HZ)
        ecg_5s         = ecg_1d[ecg_start : ecg_start + WINDOW_5S_ECG].astype(np.float64)
        ecg_5s_metrics = (
            compute_5s_ecg_metrics(ecg_5s)
            if len(ecg_5s) == WINDOW_5S_ECG
            else {"sqi": None, "nk_sqi": None, "qrs_energy": None,
                  "vital_kurtosis": None, "instant_hr": None, "n_r_peaks": 0}
        )

        # 5s ACC metrics
        acc_feats = {"mean_mag_mg": None, "var_mag_mg2": None,
                     "spectral_entropy": None, "median_freq_hz": None}
        if acc_signal is not None:
            acc_start = int(start_s * ACC_HZ)
            acc_5s    = acc_signal[acc_start : acc_start + WINDOW_5S_ACC]
            if len(acc_5s) == WINDOW_5S_ACC:
                acc_feats = compute_acc_har_features(acc_5s, fs=float(ACC_HZ))

        # 30s HRV broadcast
        hrv_window_idx = int(start_s // 30)
        hrv_metrics    = hrv_cache.get(hrv_window_idx, {
            "rmssd": None, "sdnn": None, "mean_hr": None, "lf_hf": None,
            "p_width": None, "qrs_width": None, "st_width": None,
            "qt_width": None, "qtc_width": None,
            "n_peaks_30s": None, "hrv_status": "no_window",
        })

        row = {
            "record_name"     : record_name,
            "patient_id"      : patient_id,
            "session"         : session,
            "segment_index"   : seg_idx,
            "segment_start_s" : round(start_s, 3),
            "segment_end_s"   : round(end_s, 3),
            "hrv_window_index": hrv_window_idx,
            "sample_rate"     : fs,
            "lead_names"      : ','.join(lead_names) if lead_names else '',
            "n_leads_active"  : int(np.sum([np.any(signal[i] != 0)
                                            for i in range(signal.shape[0])])),
            "exercise_label"  : _label_for_segment(start_s),
            **{f"ecg_{k}": v for k, v in ecg_5s_metrics.items()},
            **{f"acc_{k}": v for k, v in acc_feats.items()},
            **{f"hrv_{k}": v for k, v in hrv_metrics.items()},
        }

        for j, val in enumerate(embeddings[seg_idx]):
            row[f"emb_{j}"] = float(val)

        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Config ────────────────────────────────────────────────────────────────
    ECG_FOLDER      = "physionet.org/files/wearable-exercise-frailty/1.0.0/ecg"
    ACC_FOLDER      = "physionet.org/files/wearable-exercise-frailty/1.0.0/acc"
    CHECKPOINT_PATH = "ckpts/mimic_iv_ecg_physionet_pretrained.pt"
    OUTPUT_PARQUET  = "ecg_fm_lookup_table.parquet"
    CHUNK_SIZE      = 32
    NUM_WORKERS     = 4
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if not os.path.exists(CHECKPOINT_PATH):
        print("Downloading checkpoint from HuggingFace ...")
        os.makedirs("ckpts", exist_ok=True)
        hf_hub_download(
            repo_id   = "wanglab/ecg-fm",
            filename  = "mimic_iv_ecg_physionet_pretrained.pt",
            local_dir = "ckpts",
        )

    # ── Load data ─────────────────────────────────────────────────────────────
    ecg_records = load_wfdb_folder(ECG_FOLDER)[:1]
    print(f"ECG records : {len(ecg_records)}")

    acc_map = {}
    if os.path.exists(ACC_FOLDER):
        acc_map = load_acc_folder(ACC_FOLDER)
        print(f"ACC records : {len(acc_map)}")
    else:
        print("[WARN] ACC folder not found — ACC features will be null")

    # ── Schema & transforms ───────────────────────────────────────────────────
    schema = ECGInputSchema(
        sample_rate         = SAMPLE_RATE,
        expected_lead_order = ECG_FM_LEAD_ORDER,
        min_num_samples     = N_SAMPLES,
        partial_leads       = True,
    )
    transforms = build_transforms()

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading model ...")
    model, cfg, task = build_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)
    print(f"Model    : {type(model).__name__}")
    print(f"Params   : {sum(p.numel() for p in model.parameters()):,}")

    # ── Process all records ───────────────────────────────────────────────────
    all_dfs = []

    for i, record in enumerate(ecg_records):
        print(f"\n[{i+1}/{len(ecg_records)}] {record[0]}")
        try:
            df = process_record(
                record      = record,
                acc_map     = acc_map,
                model       = model,
                schema      = schema,
                transforms  = transforms,
                device      = DEVICE,
                chunk_size  = CHUNK_SIZE,
                num_workers = NUM_WORKERS,
            )
            if df is None:
                print(f"  [SKIP] no segments produced — record may be too short")
                continue
            all_dfs.append(df)
            print(f"  rows={len(df)}  emb_dim={sum(1 for c in df.columns if c.startswith('emb_'))}")
        except Exception as e:
            print(f"  [ERROR] {record[0]}: {e}")
            continue

    # ── Save ──────────────────────────────────────────────────────────────────
    if not all_dfs:
        print("\nNo records produced output — check data and preprocessing config.")
    else:
        results = pd.concat(all_dfs, ignore_index=True)

        # Parquet preferred over CSV:
        #   - native float32/float64 (no precision loss)
        #   - ~5-10x smaller than CSV for embedding columns
        #   - columnar layout lets downstream tools load only needed columns
        results.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow", compression="snappy")

        emb_cols  = [c for c in results.columns if c.startswith("emb_")]
        meta_cols = [c for c in results.columns if not c.startswith("emb_")]

        print(f"\n{'─'*60}")
        print(f"Output   : {OUTPUT_PARQUET}")
        print(f"Rows     : {len(results):,}")
        print(f"Columns  : {len(results.columns)}  "
              f"({len(meta_cols)} metadata + {len(emb_cols)} embedding dims)")
        print(f"\nMetadata columns:")
        for c in meta_cols:
            print(f"  {c}")
        print(f"\nFirst row preview:")
        print(results[meta_cols[:12]].iloc[0])
"""Inferensi S1/S2/S3: mean 5 fold -> mean 4 tissue -> mean 3 plane."""
from pathlib import Path
import gc
import hashlib
import io
import json
import re
import shutil
import tempfile
import threading
import zipfile
import numpy as np
from PIL import Image

from model_registry import SCENARIOS, model_inventory, keras_path

ROOT = Path(__file__).resolve().parent
PLANES = ('axial', 'coronal', 'sagittal')
TISSUES = ('ANAT', 'CSF', 'GM', 'WM')
EXPECTED = {(p, t) for p in PLANES for t in TISSUES}
# Nilai persis dari keluaran notebook voting, bukan threshold per-plane.
THRESHOLD = 0.45000000000000023
LOCK = threading.Lock()
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def collect_images(uploads):
    """uploads: iterable (filename, bytes). ZIP dibaca tanpa ekstraksi ke disk."""
    records = []
    for name, data in uploads:
        if len(data) > 60 * 1024 * 1024:
            raise ValueError('Batas unggahan 60 MB per berkas.')
        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                entries = [i for i in z.infolist() if not i.is_dir() and '__MACOSX' not in i.filename and not Path(i.filename).name.startswith('.')]
                if len(entries) > 40:
                    raise ValueError('ZIP berisi terlalu banyak berkas. Unggah satu subjek saja.')
                for entry in entries:
                    if not entry.filename.lower().endswith('.png'):
                        raise ValueError('ZIP citra hanya boleh memuat PNG.')
                    if entry.file_size > MAX_IMAGE_BYTES:
                        raise ValueError('Ukuran satu PNG melebihi 10 MB.')
                    records.append((Path(entry.filename).name, z.read(entry)))
        elif name.lower().endswith('.png'):
            records.append((Path(name).name, data))
        else:
            raise ValueError('Gunakan PNG atau ZIP berisi PNG.')
    if len(records) > 12:
        raise ValueError('Maksimal 12 PNG dari satu subjek.')
    result, subjects = {}, set()
    pattern = r'(?P<subject>.+)_(?P<tissue>ANAT|CSF|GM|WM)_(?P<plane>axial|coronal|sagittal)(?:_slice_\d+)?\.png'
    for name, data in records:
        match = re.fullmatch(pattern, name, re.IGNORECASE)
        if not match:
            raise ValueError(f'Nama tidak dikenali: {name}. Gunakan ID_ANAT_axial_slice_80.png atau ID_ANAT_axial.png.')
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f'PNG terlalu besar: {name}')
        with Image.open(io.BytesIO(data)) as im:
            if im.format != 'PNG' or im.width * im.height > 4_000_000:
                raise ValueError(f'PNG tidak sesuai atau terlalu besar: {name}')
            # PNG pipeline penelitian disimpan uint8; jangan menerima 16-bit diam-diam.
            if im.mode not in ('L', 'RGB'):
                raise ValueError(f'{name}: gunakan PNG grayscale/RGB 8-bit dari slicing, tanpa alpha.')
            im.verify()
        key = (match['plane'].lower(), match['tissue'].upper())
        if key in result:
            raise ValueError(f'Kombinasi duplikat: {key[1]} {key[0]}')
        result[key] = {'name': name, 'data': data}
        subjects.add(match['subject'])
    if len(subjects) > 1:
        raise ValueError('Nama berkas menunjukkan lebih dari satu subjek/pemindaian.')
    return result, next(iter(subjects), '')


def decode_image(data):
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
    image = tf.image.decode_png(data, channels=3)
    image = tf.image.resize(image, (224, 224), method='bilinear')
    image = tf.cast(image, tf.float32)
    return preprocess_input(image)  # tidak dibagi 255 lagi


def summarize(probabilities, threshold=THRESHOLD, scenario="S1"):
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (3, 5, 4) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Probabilitas harus berbentuk (3 plane, 5 fold, 4 tissue) dan berada pada 0–1.')
    # mean fold menggunakan float32 seperti np.vstack(...).mean pada notebook.
    tissue = values.astype(np.float32).mean(axis=1)
    # CSV -> pandas pada notebook membaca kembali probabilitas sebagai float64.
    plane = tissue.astype(np.float64).mean(axis=1)
    final = float(plane.mean())
    rows = [{'plane': p, 'tissue': t, 'prob_asd': float(tissue[i,j])}
            for i,p in enumerate(PLANES) for j,t in enumerate(TISSUES)]
    return {'scenario':scenario, 'threshold':threshold, 'prob_asd':final,
            'prediction':'ASD' if final >= threshold else 'TC',
            'plane_probabilities':dict(zip(PLANES, map(float, plane))),
            'tissue_probabilities':rows, 'fold_probabilities':values.tolist()}


def predict_subject(images, progress=None, scenario="S1"):
    if set(images) != EXPECTED:
        raise ValueError('Diperlukan tepat 12 kombinasi citra.')
    inventory = model_inventory(scenario)
    paths = {(r['plane'],r['fold']):r['path'] for r in inventory}
    missing = [r['file'] for r in inventory if not r['available']]
    if missing:
        raise ValueError('Model belum lengkap: ' + ', '.join(missing))
    import tensorflow as tf
    probabilities = np.zeros((3, 5, 4), dtype=np.float32)
    # Serial untuk menghindari clear_session bertabrakan antar sesi pengguna.
    with LOCK:
        for i, plane in enumerate(PLANES):
            batch = tf.stack([decode_image(images[(plane,t)]['data']) for t in TISSUES])
            for fold in range(5):
                path = paths[(plane,fold)]
                model = None
                try:
                    with keras_path(path) as usable_path:
                        model = tf.keras.models.load_model(usable_path, compile=False, safe_mode=True)
                        probabilities[i,fold] = model(batch, training=False).numpy().reshape(4)
                finally:
                    del model
                    tf.keras.backend.clear_session()
                    gc.collect()
                if progress:
                    progress((i*5+fold+1)/15, f'{plane} · fold {fold}')
    return summarize(probabilities, SCENARIOS[scenario]["threshold"], scenario)


if __name__ == '__main__':
    import argparse
    from model_registry import check_models
    parser = argparse.ArgumentParser(description='Periksa model lokal S1/S2/S3, tanpa impor ulang.')
    parser.add_argument('--check-models', choices=['S1','S2','S3','all'], required=True)
    parser.add_argument('--load', action='store_true', help='Juga muat model dan inferensi sintetis; lebih lama.')
    args = parser.parse_args()
    choices = list(SCENARIOS) if args.check_models == 'all' else [args.check_models]
    checks = [check_models(s, args.load) for s in choices]
    raise SystemExit(0 if all(checks) else 1)

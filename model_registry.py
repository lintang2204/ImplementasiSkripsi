"""Penemuan model lokal dan alias .keras sementara di drive proyek."""
from pathlib import Path
from contextlib import contextmanager
import gc
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
PLANES = ('axial','coronal','sagittal')
SCENARIOS = {
 'S1': {'label':'S1 · EfficientNetB0','prefix':'best_baseline','folder':'S1baseline_model','threshold':0.45000000000000023},
 'S2': {'label':'S2 · EfficientNetB0 + ASPP','prefix':'best_aspp','folder':'S2aspp_model','threshold':0.4100000000000002},
 'S3': {'label':'S3 · ASPP + Fine-tuning','prefix':'best_finetune_block7','folder':'S3FineTuning_model','threshold':0.4200000000000002},
}


def source_root(scenario):
    settings_path = ROOT / 'model_paths.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8-sig')) if settings_path.exists() else {}
    path = Path(settings.get(scenario, '../'+SCENARIOS[scenario]['folder']))
    return (ROOT/path).resolve() if not path.is_absolute() else path


def model_inventory(scenario='S1'):
    config = SCENARIOS[scenario]
    source = source_root(scenario)
    # S1 lama tetap kompatibel; bila models belum lengkap, gunakan folder sumber.
    old = ROOT/'models'
    if scenario=='S1' and all((old/f'best_baseline_{p}_fold_{f}.keras').is_file() for p in PLANES for f in range(5)):
        source = old
    pattern = re.compile(re.escape(config['prefix'])+r'_(axial|coronal|sagittal)_fold_([0-4])(?:\(\d+\))?\.(?:zip|keras)$', re.I)
    found={}
    if source.is_dir():
        for file in source.rglob('*'):
            if file.is_file():
                match=pattern.fullmatch(file.name)
                if match:found.setdefault((match[1].lower(),int(match[2])),[]).append(file)
    rows=[]
    for p in PLANES:
        for f in range(5):
            files=found.get((p,f),[])
            rows.append({'plane':p,'fold':f,'file':f'{config["prefix"]}_{p}_fold_{f}.keras',
                         'available':len(files)==1,'path':str(files[0]) if len(files)==1 else '',
                         'status':'OK' if len(files)==1 else ('DUPLIKAT' if files else 'TIDAK ADA')})
    return rows


@contextmanager
def keras_path(source):
    source=Path(source)
    if source.suffix.lower()=='.keras':
        yield source
        return
    # Semua berkas kerja di lokasi aplikasi (E: pada laptop pengguna), bukan TEMP C:.
    work=ROOT/'.model_work'
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as temp:
        dest=Path(temp)/'model.keras'
        with zipfile.ZipFile(source) as z:
            direct={'metadata.json','config.json','model.weights.h5'} <= set(z.namelist())
            if not direct:
                entries=[i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith('.keras')]
                if len(entries)!=1:raise ValueError('ZIP pembungkus harus berisi tepat satu .keras.')
                if entries[0].file_size>1024**3:raise ValueError('Model melebihi batas 1 GiB.')
                with z.open(entries[0]) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
        if direct:
            try:os.link(source,dest)  # alias tanpa menggandakan bobot pada filesystem yang mendukung
            except OSError:shutil.copyfile(source,dest)
        yield dest


def inspect_archive(path, scenario, plane):
    with zipfile.ZipFile(path) as z:
        if not {'metadata.json','config.json','model.weights.h5'} <= set(z.namelist()):raise ValueError('Arsip model tidak lengkap.')
        if z.testzip():raise ValueError('CRC arsip gagal.')
        conf=json.loads(z.read('config.json'))['config']
        expected=('Baseline_EfficientNetB0_' if scenario=='S1' else 'EfficientNetB0_ASPP_')+plane
        if conf['name']!=expected:raise ValueError(f'Nama model {conf["name"]}, diharapkan {expected}.')
        layers=conf['layers']
        if [l['config'].get('batch_shape') for l in layers if l['class_name']=='InputLayer'] != [[None,224,224,3]]:raise ValueError('Input bukan 224×224×3.')
        if layers[-1]['config'].get('units')!=1 or layers[-1]['config'].get('activation')!='sigmoid':raise ValueError('Output bukan sigmoid tunggal.')
        aspp={l['config']['name']:l['config'] for l in layers if l['config']['name'].startswith('aspp_conv_rate')}
        if scenario!='S1':
            for rate in (6,12,18):
                if aspp.get(f'aspp_conv_rate{rate}',{}).get('dilation_rate')!=[rate,rate]:raise ValueError('Cabang ASPP tidak sesuai notebook.')
            block=[l for l in layers if l['config']['name'].startswith('block7a_') and l['class_name']=='Conv2D']
            if not block or any(bool(l['config'].get('trainable'))!=(scenario=='S3') for l in block):raise ValueError('Pengaturan trainable block7a tidak sesuai S2/S3; cek model sumber.')
        digest=hashlib.sha256()
        with z.open('model.weights.h5') as weights:
            for chunk in iter(lambda:weights.read(1024*1024),b''):digest.update(chunk)
        return digest.hexdigest()


def check_models(scenario, run_load=False):
    rows=model_inventory(scenario);seen=set();passed=0
    for row in rows:
        title=f'{scenario} {row["plane"]} fold {row["fold"]}'
        if not row['available']:
            print(title+': '+row['status'],flush=True);continue
        try:
            with keras_path(row['path']) as path:
                digest=inspect_archive(path,scenario,row['plane'])
                if digest in seen:raise ValueError('Bobot identik dengan model sebelumnya pada skenario ini.')
                seen.add(digest)
                if run_load:
                    import tensorflow as tf
                    model=None
                    try:
                        model=tf.keras.models.load_model(path,compile=False,safe_mode=True)
                        value=model(tf.zeros((1,224,224,3)),training=False).numpy()
                        import numpy as np
                        if value.shape!=(1,1) or not np.isfinite(value).all() or not ((value>=0)&(value<=1)).all():raise ValueError('Keluaran uji sintetis tidak valid.')
                    finally:
                        del model
                        tf.keras.backend.clear_session();gc.collect()
            passed+=1
            print(title+(': ARSIP + INFERENSI SINTETIS OK' if run_load else ': ARSIP OK'),flush=True)
        except Exception as exc:print(title+': GAGAL — '+str(exc),flush=True)
    print(f'{scenario}: {passed}/15 lolos'+(' uji muat/inferensi sintetis' if run_load else ' pemeriksaan arsip'),flush=True)
    return passed==15

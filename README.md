# Streamlit ASD–TC — S1

Aplikasi lokal untuk 15 model EfficientNetB0: 3 bidang potong × 5 fold.
Bahasa antarmuka: Indonesia. Tidak melakukan training ulang.

## Isi paket

- app.py: antarmuka, unggahan, preview, hasil dan ekspor.
- inference.py: validasi, impor model, pemrosesan dan voting.
- requirements.txt: versi yang diuji pada Python 3.12/Linux CPU.
- test_app.py: pemeriksaan input dan matematika voting.
- model_manifest.json: nama dan checksum 15 model yang sudah diperiksa.
- VALIDASI.md: cakupan pengujian dan batas hasil.

Berkas bobot tidak digandakan dalam ZIP kode agar unduhan ringan. Gunakan 15 ZIP
baseline yang sudah Anda unduh. Paket ini BELUM memuat S2 dan S3.

## Instalasi Windows (Command Prompt/CMD)

Gunakan Python 3.12 64-bit. Ekstrak paket, buka CMD di folder yang memuat app.py.

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Simpan 15 ZIP model di satu folder, misalnya C:\model_baseline. Folder boleh
memiliki subfolder. Impor model:

```bat
python inference.py --import-models "C:\model_baseline"
```

Perintah memeriksa arsip dan menyalin model ke folder models dengan ekstensi
.keras. ZIP langsung berisi config.json/metadata.json/model.weights.h5 dan ZIP
pembungkus berkas .keras didukung. Fold 0(1) juga dikenali. Duplikat identik
hanya dipakai satu kali; duplikat berbeda untuk fold sama ditolak. Berkas sumber
tidak diubah. Jangan arahkan impor ke folder berisi ZIP citra atau file unrelated.

Jalankan:

```bat
python -m streamlit run app.py
```

Buka alamat lokal yang ditampilkan terminal (biasanya http://localhost:8501).
Hentikan dengan Ctrl+C. Windows belum diuji langsung; jika instalasi gagal,
simpan pesan error untuk menyesuaikan lingkungan.

## Masukan

12 PNG 8-bit hasil slicing penelitian: ANAT/CSF/GM/WM × axial/coronal/sagittal,
seluruhnya satu subjek/pemindaian. Unggah sekaligus atau ZIP berisi PNG tersebut.

Nama asli slicing didukung:
`ID_ANAT_axial_slice_80.png`
Nama singkat juga diterima: `ID_ANAT_axial.png`.

Awalan ID harus konsisten. Contoh lengkap (angka slice boleh berbeda per plane):

```
ID_ANAT_axial.png       ID_CSF_axial.png
ID_GM_axial.png         ID_WM_axial.png
ID_ANAT_coronal.png     ID_CSF_coronal.png
ID_GM_coronal.png       ID_WM_coronal.png
ID_ANAT_sagittal.png    ID_CSF_sagittal.png
ID_GM_sagittal.png      ID_WM_sagittal.png
```

Pemeriksaan nama tidak membuktikan identitas subjek dari anatomi. Pengguna tetap
harus memastikan asal citra. PNG alpha/16-bit ditolak agar tidak diam-diam
mengubah input. Masukan bukan MRI mentah, JPEG, screenshot, atau hasil resize
manual dari penampil gambar. Aplikasi tidak melakukan skull stripping/segmentasi.

## Alur inferensi

1. tf.image.decode_png(channels=3), resize bilinear 224×224, float32,
   tensorflow.keras.applications.efficientnet.preprocess_input.
2. Tidak ada augmentasi; tidak membagi masukan dengan 255 lagi.
3. Lima model fold memprediksi empat citra pada masing-masing plane.
4. Mean lima probabilitas model untuk setiap citra, lalu mean empat representasi.
5. Mean probabilitas axial/coronal/sagittal.
6. ASD bila skor >= 0.45000000000000023, selain itu TC. Nilai ini disalin persis
   dari output threshold final OOF S1 pada cobavoting (1).ipynb; tampilan dibulatkan 0,45.

Model dimuat satu per satu dengan compile=False, safe_mode=True, training=False.
Bobot tidak berubah. Penggunaan memori dibatasi melalui pelepasan tiap model;
prediksi pertama memerlukan waktu dan kecepatannya bergantung pada CPU/RAM.
Tidak ada training atau threshold tuning saat pengguna mengunggah gambar.

## Pengujian

```bat
python -m unittest test_app.py
```

Untuk verifikasi penelitian yang belum selesai, gunakan 12 citra subjek uji yang
sama dengan notebook dan bandingkan probabilitas tiap plane serta probabilitas
akhir dari JSON aplikasi dengan CSV notebook. Jangan hanya membandingkan kelas.
Perbedaan numerik kecil antarplatform mungkin terjadi; selisih yang bermakna perlu
diselidiki. Uji sintetis bukan validasi akurasi atau diagnosis.

## Data dan penggunaan

Gambar unggahan dibaca dalam memori aplikasi, tidak ditulis oleh kode ke disk.
Hasil tetap dalam sesi hingga input berubah/sesi berakhir dan dapat diunduh CSV/JSON.
Jalankan lokal untuk tahap awal. Aplikasi belum dipublikasikan daring.
Hasil adalah prediksi penelitian, bukan diagnosis klinis atau probabilitas klinis
terkalibrasi. Accuracy eksperimen tidak dihitung ulang dari satu subjek.

## Acuan API

- https://keras.io/api/models/model_saving_apis/model_saving_and_loading/
- https://docs.streamlit.io/develop/api-reference/widgets/st.download_button
- https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest

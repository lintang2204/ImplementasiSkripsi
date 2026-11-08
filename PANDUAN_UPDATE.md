# Pembaruan S1–S2–S3 (tanpa upload bobot)

1. Hentikan Streamlit di terminal dengan Ctrl+C.
2. Cadangkan app.py dan inference.py lama, lalu salin SELURUH isi folder update_s123
   ke folder streamlit_asd_tc yang sudah ada, sejajar dengan requirements.txt.
   Timpa app.py dan inference.py; tambahkan model_registry.py dan model_paths.json.
   Jangan membuat folder bersarang. Jangan hapus .venv atau models.
3. Tidak ada dependensi baru. Lingkungan dari aplikasi S1 tetap dipakai.
4. Dari PowerShell pada folder streamlit_asd_tc:

```powershell
.\.venv\Scripts\python.exe inference.py --check-models all
```

Ini memeriksa arsip, nama model, cabang ASPP, pengaturan trainable block7a S2/S3,
input, output, serta bobot duplikat. Status 15/15 di UI hanya kelengkapan nama file,
bukan bukti keberhasilan pemuatan model. Fold didasarkan nama file; tidak ada
metadata fold independen dalam arsip untuk membuktikan asal pembagian data.

Uji pemuatan dan inferensi sintetis model lokal (lebih lama):

```powershell
.\.venv\Scripts\python.exe inference.py --check-models S2 --load
.\.venv\Scripts\python.exe inference.py --check-models S3 --load
```

Lalu jalankan aplikasi:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Pilih S1/S2/S3 di sidebar. Unggah 12 PNG subjek sama. Ambang final OOF:
S1=0.45000000000000023, S2=0.4100000000000002, S3=0.4200000000000002.
Saat mengganti skenario, hasil skenario sebelumnya tidak ditampilkan sebagai hasil baru.
Ekspor CSV/JSON mencantumkan skenario. Model tidak dilatih ulang.

## Folder yang dibaca

Default relatif dari folder aplikasi:
- ../S1baseline_model
- ../S2aspp_model
- ../S3FineTuning_model

Subfolder axial/coronal/sagittal otomatis dipindai. S1 menggunakan folder models
lama jika seluruh 15 .keras tersedia di sana. S2/S3 tidak perlu diimpor atau diubah
namanya. Sesuaikan model_paths.json jika folder berpindah. JSON Windows memakai
slash / atau backslash ganda. Hindari duplikat bernomor (1) dalam folder sumber;
pindahkan salinan ekstra ke folder lain di luar folder sumber, jangan hapus asli.

## Ruang disk

Arsip ZIP langsung adalah bentuk Keras yang sama dengan ekstensi berbeda.
Kode mencoba hard link sementara berekstensi .keras, tanpa menduplikasi bobot.
Jika tidak didukung filesystem, kode menyalin hanya satu model sementara.
ZIP pembungkus satu .keras juga didukung. Berkas sementara ditempatkan di
streamlit_asd_tc/.model_work pada drive proyek (E:), dihapus saat operasi selesai.
Tidak ada ekstraksi ke C: oleh kode ini. Windows/TensorFlow sendiri tetap dapat
memakai sumber daya sistem. Jangan menghapus folder kerja saat prediksi berjalan.

## Validasi versi pembaruan

Diuji: 5 arsip S2 axial yang diberikan (CRC, arsitektur, bobot berbeda), deteksi
S2 yang salah ditempatkan sebagai S3, penemuan 15 nama per skenario di folder uji,
penolakan duplikat, enam tes input/voting versi awal, pemilihan threshold S1/S2/S3.
Tidak seluruh 30 model S2/S3 tersedia di lingkungan pengembang. Uji pemuatan lokal
--load diperlukan; jika gagal kirim teks log, tidak perlu mengirim semua bobot.
Belum ada validasi numerik terhadap PNG subjek nyata dan CSV hasil notebook.
Model S3 dimuat dari checkpoint S2 dalam notebook, sehingga nama internal model
masih EfficientNetB0_ASPP_<plane>; flag trainable block7a dipakai sebagai pemeriksaan
konfigurasi tambahan, bukan bukti bahwa proses training berjalan dengan benar.

README lama menjelaskan aplikasi S1; panduan ini menggantikan bagian cakupan,
impor model, dan cara pemeriksaan tersebut. Jangan jalankan --import-models lama.

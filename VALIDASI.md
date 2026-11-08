# Catatan validasi versi awal

- 15 arsip baseline (axial, coronal, sagittal × fold 0–4) lengkap.
- Semua lolos CRC dan pemeriksaan nama, bentuk input serta output.
- Semua berhasil dimuat Keras dengan compile=False dan safe_mode=True.
- Uji inferensi memakai PNG sintetis konstan 224×224 berhasil melewati seluruh
  model, agregasi lima fold, empat representasi dan tiga plane. Durasi satu
  pengujian fungsi sekitar 29 detik di lingkungan ini, bukan janji kinerja laptop.
- Enam tes validasi/matematika lulus: 12 kombinasi, ZIP citra, duplikasi,
  subjek bercampur, agregasi, probabilitas tidak valid.
- Streamlit AppTest halaman awal dan alur unggahan sintetis → konfirmasi →
  prediksi 15 model → hasil serta tombol ekspor berjalan tanpa exception.

Belum selesai: pembandingan probabilitas pada subjek nyata dengan CSV notebook,
pengujian langsung Windows, pemeriksaan tampilan secara visual di browser,
dan deployment daring. Tidak ada klaim akurasi atau validitas klinis dari tes ini.

Lingkungan: Python 3.12, tensorflow-cpu 2.20.0, Keras 3.13.2, Streamlit 1.50.0,
NumPy 2.3.5, pandas 2.2.3, Pillow 11.3.0.

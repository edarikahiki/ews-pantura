# EWS Pantura

Repositori ini berisi notebook dan kode Python untuk menyiapkan serta menjalankan pemodelan banjir pesisir dengan SFINCS. Eksekusi model dilakukan menggunakan executable SFINCS yang disertakan di repositori.

## Instalasi dependensi

Gunakan Miniforge atau Miniconda dengan akses ke `conda-forge`. Dari direktori utama repositori, buat dan aktifkan environment berikut:

```powershell
conda env create -f env.yml
conda activate multi-hazard
```

Environment `multi-hazard` menggunakan Python 3.11. Berkas `env.yml` mencantumkan dependensi Python, Jupyter, komputasi ilmiah dan geospasial, serta paket HydroMT/SFINCS. Jika environment sudah pernah dibuat dan perlu diperbarui setelah `env.yml` berubah, jalankan:

```powershell
conda env update -f env.yml --prune
```

Pastikan environment aktif sebelum menjalankan notebook atau skrip Python.

## Struktur repositori

- `data/geojson/` — tempat yang diharapkan untuk data vektor masukan, misalnya batas wilayah atau fitur pesisir dalam format GeoJSON.
- `data/tiff/` — tempat yang diharapkan untuk data raster masukan, misalnya elevasi, batimetri, atau penggunaan lahan dalam format GeoTIFF.
- `notebook/` — notebook Jupyter untuk eksperimen dan alur kerja pemodelan. Notebook yang saat ini tersedia adalah `trial_model_online.ipynb`.
- `python/` — modul Python yang digunakan notebook dan alur pemodelan, termasuk pembungkus model, pengolahan batas muka air, dan fungsi analisis hidrogaraf.
- `SFINCS_v2.3.0_mt_Faber_release_exe/` — executable SFINCS beserta pustaka runtime dan berkas lisensinya.

## Berkas di direktori utama

- `env.yml` — definisi environment Conda dan dependensi proyek.
- `data_catalog.yml` — katalog sumber data raster yang digunakan alur kerja. Periksa dan sesuaikan lokasi data pada katalog dengan lokasi data di komputer Anda sebelum menjalankan alur yang menggunakannya.
- `run.bat` — skrip Windows untuk menjalankan executable SFINCS dari folder repositori dan menyimpan keluaran konsol ke `sfincs_log.txt`.

# SportVision — Deteksi Aktivitas Manusia untuk Olahraga

Aplikasi Computer Vision untuk mendeteksi aktivitas manusia pada video olahraga. Setiap manusia yang teramati diberi bounding box hijau pada frame hasil analisis.

Output sistem adalah **deteksi**, bukan klasifikasi tampilan: aplikasi menunjukkan lokasi subjek manusia dan ringkasan frame yang memuat aktivitas manusia. Sebelum deteksi, video divalidasi menggunakan model yang dilatih dari dataset proyek. Hanya video yang cukup cocok dengan kategori `CricketShot`, `Punch`, atau `TennisSwing` yang akan menampilkan hasil deteksi. Video di luar cakupan dataset tidak menampilkan bounding box.

## Menjalankan aplikasi

1. Aktifkan virtual environment yang digunakan proyek.
2. Instal dependensi bila belum tersedia:

   ```powershell
   pip install -r requirements.txt
   ```

3. Jalankan dashboard:

   ```powershell
   streamlit run 08_web_app.py
   ```

## Struktur utama

- `08_web_app.py` — dashboard Streamlit dan pipeline deteksi manusia.
- `Dataset/test/` — video contoh untuk pengujian.

Video diproses pada delapan frame yang tersebar merata. Detektor OpenCV HOG menandai lokasi manusia dalam setiap frame dengan bounding box.

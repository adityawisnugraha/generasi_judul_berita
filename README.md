
# Generasi Judul Berita

Proyek ini bertujuan untuk membangun model pembuat judul berita berdasarkan isi artikel menggunakan tiga pendekatan arsitektur berbeda: LSTM Encoder-Decoder, LSTM Encoder-Decoder + Attention, dan Transformer.

## Pendahuluan

Dalam proyek ini, dilakukan eksperimen untuk membangun model yang mampu menghasilkan judul berita dari konten artikel menggunakan tiga pendekatan arsitektur yang berbeda: (1) LSTM Encoder-Decoder, (2) LSTM Encoder-Decoder dengan Attention, dan (3) Transformer.

## Temuan Eksperimen

**LSTM Encoder-Decoder**
- BLEU Score: rendah.
- Akurasi konten rendah.
- Kesulitan menangkap kata kunci penting.

**LSTM Encoder-Decoder dengan Attention**
- BLEU Score: meningkat dibanding LSTM biasa.
- Lebih relevan dan fokus terhadap bagian penting artikel.

**Transformer**
- BLEU Score: tertinggi.
- Menghasilkan judul variatif, kreatif, dan sangat relevan.

## Perbandingan Arsitektur

| Aspek                    | LSTM Encoder-Decoder | LSTM + Attention | Transformer |
|---------------------------|-----------------------|------------------|-------------|
| Akurasi Konten            | Rendah                | Sedang           | Tinggi      |
| Relevansi Kata Kunci      | Sering terlewat        | Cukup baik       | Sangat baik |
| Kompleksitas Model        | Rendah                | Sedang           | Tinggi      |
| Waktu Pelatihan           | Cepat                 | Lebih lama       | Paling lama |
| Generalisasi ke Berita Baru| Kurang baik           | Cukup baik       | Sangat baik |

## Kesimpulan

Kompleksitas tambahan seperti Attention dan Transformer secara signifikan meningkatkan performa. Transformer terbukti menjadi pilihan utama untuk tugas generasi judul berita ini.

## Instalasi

Clone repo ini dan instal dependensi:

```bash
git clone https://github.com/adityawisnugraha/generasi_judul_berita.git
cd generasi_judul_berita
pip install -r requirements.txt
```

## Menjalankan Notebook atau Script

Buka `Generate_Judul_Berita.ipynb` menggunakan Jupyter Notebook atau jalankan script Python:

```bash
python generate_judul_berita.py
```

# Layout Aware Text Extraction (Computer Vision)

Proyek ini adalah sebuah *pipeline* berbasis Computer Vision (CV) dan Optical Character Recognition (OCR) yang berfungsi untuk mengekstrak teks dari sebuah gambar (seperti *slide* presentasi, poster, atau dokumen) dengan mempertahankan tata letak (*layout*), gaya tulisan (ukuran, ketebalan, warna, dan perataan), serta menghapus teks dari gambar aslinya (menghasilkan *background* bersih).

Hasil akhirnya adalah sebuah halaman HTML interaktif di mana *background* gambar yang telah dibersihkan ditampilkan di latar belakang, dan teks yang diekstrak di-render ulang menggunakan elemen HTML (teks biasa yang bisa di-blok/copy) dengan penempatan (*absolute positioning*) dan *styling* yang menyerupai aslinya.

## ✨ Fitur Utama

1. **Deteksi Teks Akurat**: Menggunakan `EasyOCR` dengan dukungan bahasa Inggris (`en`) dan bahasa Indonesia (`id`).
2. **Ekstraksi Atribut Visual Teks**:
   - **Warna Teks**: Menggunakan algoritma *K-Means Clustering* untuk secara cerdas memisahkan warna teks murni dari *background*, menghindari kontaminasi *anti-aliasing* atau *shadow*.
   - **Ukuran Font (Font Size)**: Estimasi ukuran *font* secara dinamis berdasarkan tinggi *bounding box* hasil OCR.
   - **Ketebalan Teks (Font Weight)**: Deteksi otomatis teks *bold* vs *normal* dengan menganalisis ketebalan guratan (*stroke*) via `distance transform`.
3. **Pengelompokan Multiline (Text Grouping)**: Menggabungkan baris-baris teks yang berdekatan secara vertikal dan menumpuk secara horizontal menjadi satu paragraf utuh (*multiline*). Algoritma ini juga secara otomatis menentukan *text-alignment* (kiri atau tengah) dan *line-height*.
4. **Penghapusan Teks dari Gambar (Inpainting)**: Membuat *pixel-perfect mask* dari lokasi teks dan menggunakan metode *Navier-Stokes* via `OpenCV` untuk "menambal" area teks dengan *background* di sekitarnya.
5. **Output HTML Interaktif**: Menggunakan `Jinja2` untuk men-generate file HTML. Anda mendapatkan *file* gambar latar belakang tanpa teks dan *file* HTML yang menggabungkan *background* tersebut dengan teks interaktif di atasnya.

## 🛠️ Prasyarat (Requirements)

Pastikan Python telah terinstal di sistem Anda (Python 3.8+ disarankan).

Library atau dependensi utama yang dibutuhkan adalah:
- `easyocr` (Untuk deteksi dan pengenalan teks)
- `opencv-python-headless` / `cv2` (Pemrosesan gambar, pembuatan *mask*, dan *inpainting*)
- `numpy` (Operasi matriks dan array)
- `scikit-learn` (Algoritma *K-Means* untuk ekstraksi warna teks)
- `jinja2` (Sistem *templating* untuk membuat file HTML)

Semua dependensi terdaftar dalam file `requirements.txt`.

## ⚙️ Instalasi

1. **Clone/Download** *repository* atau proyek ini ke komputer Anda.
2. Buka terminal atau Command Prompt dan navigasikan ke direktori proyek:
   ```bash
   cd "path/to/Layout Aware Text Extraction (Computer Vision)"
   ```
3. (Opsional namun disarankan) Buat dan aktifkan *virtual environment*:
   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
4. Install dependensi proyek:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Cara Penggunaan

Gunakan *script* `main.py` dari terminal dengan memberikan lokasi *file* gambar yang ingin diproses melalui argumen `--image`.

**Sintaks Dasar:**
```bash
python main.py --image <path_ke_gambar> [--output_dir <path_folder_output>]
```

**Contoh:**
```bash
# Jika gambar ada di folder 'inputs' bernama 'slide1.png'
python main.py --image inputs/slide1.png

# Anda juga bisa menentukan direktori output kustom (default: folder 'outputs')
python main.py --image inputs/slide1.png --output_dir hasil_ekstraksi
```

**Proses yang berjalan:**
1. EasyOCR akan membaca gambar. (Saat pertama kali jalan, EasyOCR mungkin akan mengunduh model model *detection* dan *recognition*).
2. Sistem akan mengekstrak detail (warna, ketebalan) dari tiap kata/kalimat.
3. *Grouping* atau pengelompokan kalimat yang berdekatan.
4. *Inpainting* (pembersihan gambar dari teks).
5. File luaran disimpan di folder `outputs` (atau folder kustom Anda).

**Hasil Output (dalam folder tujuan):**
- `<nama_gambar>_bg.jpg`: Gambar asli namun teksnya sudah dihapus oleh sistem (*inpainting*).
- `<nama_gambar>_interactive.html`: Halaman web hasil akhir (*double-click* untuk membukanya di *browser* Anda).

## 📂 Struktur Proyek

```
Layout Aware Text Extraction (Computer Vision)/
│
├── inputs/               # (Opsional) Folder untuk menyimpan gambar input.
├── outputs/              # Direktori default tempat gambar hasil inpaint & HTML disimpan.
├── templates/
│   └── template.html     # Template Jinja2 dasar untuk membentuk HTML Output.
├── main.py               # Script utama berisi core logic (OCR, KMeans, Inpaint, HTML Render).
├── requirements.txt      # Daftar pustaka atau dependensi proyek.
└── .gitignore            # File untuk mengabaikan direktori/file tertentu oleh Git.
```

## 🧠 Cara Kerja Sistem (Under the Hood)

1. **Text Detection**: Gambar dibaca oleh `cv2` lalu diumpankan ke `easyocr.Reader` untuk mengambil teks beserta *bounding box* (koordinat x dan y) dan *confidence level*.
2. **Color Extraction & Masking (`get_text_color_and_mask`)**:
   Untuk setiap *bounding box* teks:
   - Area tersebut di-*crop*.
   - Menggunakan algoritma *K-Means Clustering* (`k=2`) memisahkan piksel teks dan piksel *background*.
   - Fitur cerdas: mengecek *brightness* piksel teks dan mengambil nilai median dari "core pixels" agar mendapat warna murni tanpa terdistorsi *shadow*/*anti-aliasing*.
   - Membuat gambar *mask hitam-putih* secara akurat sesuai posisi piksel teks tersebut.
3. **Font Weight Detection (`detect_font_weight`)**:
   *Distance transform* digunakan pada *masking* hitam putih untuk mengukur setengah lebar dari ketebalan guratan teks. Jika rasionya terhadap tinggi teks melebihi ambang batas, teks dikategorikan sebagai `'bold'`, bila tidak maka `'normal'`.
4. **Multiline Grouping (`group_text_items`)**:
   Menggunakan algoritma pengelompokan yang cerdas berbasis `Union-Find`. Elemen teks dicek jika selisih jarak vertikal (Y) sangat dekat dan bertumpang tindih (X), lalu dikombinasikan menjadi 1 blok teks multiline (dipisahkan dengan enter `\n`). Proses ini juga menganalisis dan menentukan perataan teks (*left* atau *center*).
5. **Background Inpainting**:
   Semua *mask* dari masing-masing kalimat digabungkan. `cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)` dijalankan untuk menghapus semua karakter dari gambar sesuai *masking* yang ada.
6. **HTML Templating**:
   Data *array* JSON Python berisi *(x, y, w, h, text, color, font_size, weight, align, line_height)* dilempar ke `Jinja2` (template `template.html`) untuk menyusun DOM *Absolute Element* layaknya merekonstruksi *layout* asli, di atas CSS *background image* yang sudah bersih.

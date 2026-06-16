# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import easyocr
from sklearn.cluster import KMeans
# pyrefly: ignore [missing-import]
from jinja2 import Environment, FileSystemLoader
import os
import argparse
from collections import Counter


def get_text_color_and_mask(image):
    """
    Ekstraksi warna teks dan pembuatan pixel-mask yang akurat.
    
    Perbaikan dari versi sebelumnya:
    - Menggunakan MEDIAN dari pixel inti cluster teks (bukan cluster center KMeans)
      agar warna tidak terkontaminasi oleh anti-aliasing, shadow, dan gradient.
    - Mengambil 30% pixel paling representatif (brightest untuk teks terang,
      darkest untuk teks gelap) untuk mendapatkan warna murni.
    - Snap ke putih murni (255,255,255) jika brightness > 220, karena hampir pasti
      teks putih di atas background gelap.
    - Dilation kernel diperbesar untuk mask yang lebih menyeluruh.
    """
    # Resize gambar jika terlalu besar untuk mempercepat performa KMeans
    max_dim = 100
    h_orig, w_orig = image.shape[:2]
    if h_orig > max_dim or w_orig > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        image_resized = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)))
    else:
        image_resized = image

    curr_h, curr_w = image_resized.shape[:2]
    pixels = image_resized.reshape((-1, 3))
    
    # Jika gambar terlalu kecil (misalnya hanya 1 pixel)
    if len(pixels) < 2:
        mask = np.ones((h_orig, w_orig), dtype=np.uint8) * 255
        if len(pixels) == 1:
            b, g, r = [int(c) for c in pixels[0]]
            return f"rgb({r}, {g}, {b})", mask
        return "rgb(0, 0, 0)", mask
        
    try:
        # Gunakan K-Means untuk membagi 2 warna utama: Teks dan Background area
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Reshape labels kembali ke dimensi gambar (setelah resize) untuk mendeteksi tepi
        labels = kmeans.labels_.reshape((curr_h, curr_w))
        
        # Karena kita menggunakan padding 25% di setiap sisi (bawah, atas, kiri, kanan),
        # area background dipastikan mendominasi keseluruhan gambar hasil crop.
        # Oleh karena itu, warna background adalah cluster yang paling banyak muncul di seluruh area.
        counts = Counter(labels.flatten())
        bg_cluster = counts.most_common(1)[0][0]
        
        # Warna teks adalah cluster lawannya (yang bukan background)
        text_cluster = 1 if bg_cluster == 0 else 0
        
        bg_color = kmeans.cluster_centers_[bg_cluster]
        
        # --- PERBAIKAN WARNA: Median dari pixel inti teks ---
        # Alih-alih menggunakan cluster center (mean), kita ambil median dari
        # pixel-pixel yang paling representatif di cluster teks.
        text_pixel_mask = kmeans.labels_ == text_cluster
        text_pixels = pixels[text_pixel_mask]
        
        if len(text_pixels) > 5:
            # Hitung brightness setiap pixel teks (BGR format OpenCV)
            brightness_vals = (0.299 * text_pixels[:, 2] + 
                             0.587 * text_pixels[:, 1] + 
                             0.114 * text_pixels[:, 0])
            
            bg_brightness = (0.299 * bg_color[2] + 
                           0.587 * bg_color[1] + 
                           0.114 * bg_color[0])
            
            median_brightness = np.median(brightness_vals)
            
            if median_brightness > bg_brightness:
                # Teks TERANG di atas background GELAP:
                # Ambil 30% pixel PALING TERANG (mengecualikan shadow & anti-aliasing)
                threshold = np.percentile(brightness_vals, 70)
                core_pixels = text_pixels[brightness_vals >= threshold]
            else:
                # Teks GELAP di atas background TERANG:
                # Ambil 30% pixel PALING GELAP (mengecualikan highlight & anti-aliasing)
                threshold = np.percentile(brightness_vals, 30)
                core_pixels = text_pixels[brightness_vals <= threshold]
            
            if len(core_pixels) > 0:
                text_color = np.median(core_pixels, axis=0)
            else:
                text_color = np.median(text_pixels, axis=0)
        else:
            text_color = kmeans.cluster_centers_[text_cluster]
        
        # --- SNAP KE PUTIH ---
        # Jika brightness akhir > 220, teks hampir pasti putih.
        # Snap ke pure white untuk menghindari warna off-white/abu-abu.
        final_brightness = (0.299 * text_color[2] + 
                          0.587 * text_color[1] + 
                          0.114 * text_color[0])
        if final_brightness > 220:
            text_color_display = np.array([255, 255, 255], dtype=np.float64)
        else:
            text_color_display = text_color
        
        # --- BIKIN MASK AKURAT PADA RESOLUSI ASLI ---
        # Gunakan cluster center (bukan median) untuk mask karena lebih stabil secara spasial
        img_float = image.astype(np.float32)
        dist_to_text = np.sum((img_float - kmeans.cluster_centers_[text_cluster])**2, axis=2)
        dist_to_bg = np.sum((img_float - bg_color)**2, axis=2)
        
        # Pixel lebih dekat ke warna teks adalah teks
        full_mask = (dist_to_text < dist_to_bg).astype(np.uint8) * 255
        
        text_gray = (0.299 * kmeans.cluster_centers_[text_cluster][2] + 
                    0.587 * kmeans.cluster_centers_[text_cluster][1] + 
                    0.114 * kmeans.cluster_centers_[text_cluster][0])
        bg_gray = (0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0])
        
        # Perlebar (dilate) secara dinamis berdasarkan tinggi teks
        # Diperbesar dari versi sebelumnya untuk menghapus anti-aliasing & shadow lebih tuntas
        if text_gray > bg_gray:
            k_size = max(5, int(h_orig * 0.15))   # Sebelumnya 0.12
        else:
            k_size = max(3, int(h_orig * 0.08))    # Sebelumnya 0.05
            
        if k_size % 2 == 0: 
            k_size += 1
            
        kernel = np.ones((k_size, k_size), np.uint8)
        full_mask = cv2.dilate(full_mask, kernel, iterations=1)
            
        # Konversi BGR (format cv2) ke RGB untuk HTML
        b, g, r = [int(c) for c in text_color_display]
        return f"rgb({r}, {g}, {b})", full_mask
    except Exception as e:
        print(f"Peringatan: Ekstraksi warna gagal, fallback ke hitam. Error: {e}")
        return "rgb(0, 0, 0)", np.ones((h_orig, w_orig), dtype=np.uint8) * 255


def detect_font_weight(mask_region):
    """
    Deteksi apakah teks bold berdasarkan analisis ketebalan stroke.
    
    Menggunakan distance transform pada binary mask teks:
    - Jarak maksimum pada distance transform = setengah ketebalan stroke terlebar
    - Rasio (max_distance / tinggi_bbox) > 0.07 → bold
    
    Ini bersifat generik dan tidak bergantung pada font tertentu.
    """
    if mask_region is None or mask_region.size == 0:
        return 'normal'
    
    h = mask_region.shape[0]
    if h < 5:
        return 'normal'
    
    # Pastikan mask binary
    _, binary = cv2.threshold(mask_region, 127, 255, cv2.THRESH_BINARY)
    
    # Cek apakah ada pixel teks sama sekali
    if np.sum(binary) == 0:
        return 'normal'
    
    # Distance transform: setiap pixel teks diberi nilai = jarak ke pixel non-teks terdekat
    # Nilai maksimum = setengah ketebalan stroke terlebar
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_dist = np.max(dist)
    
    # Normalisasi berdasarkan tinggi bounding box
    stroke_ratio = max_dist / h
    
    # Bold text umumnya memiliki stroke_ratio > 0.11 (dinaikkan dari 0.07 agar tidak over-bold)
    if stroke_ratio > 0.11:
        return 'bold'
    return 'normal'


def group_text_items(items):
    """
    Menggabungkan item teks yang secara vertikal berdekatan dan horizontal
    overlap menjadi blok teks multiline.
    
    Algoritma:
    1. Untuk setiap pasangan item, cek apakah mereka harus digabung:
       - Jarak vertikal < 50% rata-rata tinggi kedua item
       - Overlap horizontal > 30% dari lebar item terkecil
    2. Gunakan Union-Find untuk menangani penggabungan transitif
       (A dekat B, B dekat C → A, B, C jadi satu grup)
    3. Untuk setiap grup, hitung bounding box gabungan, alignment, line-height
    """
    n = len(items)
    if n <= 1:
        return items
    
    # --- Union-Find ---
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # Path compression
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    
    # Cek semua pasangan item
    for i in range(n):
        for j in range(i + 1, n):
            item_i = items[i]
            item_j = items[j]
            
            # --- Cek jarak vertikal ---
            i_top, i_bottom = item_i['y'], item_i['y'] + item_i['h']
            j_top, j_bottom = item_j['y'], item_j['y'] + item_j['h']
            
            avg_h = (item_i['h'] + item_j['h']) / 2
            
            # Hitung gap: jarak antara bawah item atas dan atas item bawah
            if i_top < j_top:
                vertical_gap = j_top - i_bottom
            else:
                vertical_gap = i_top - j_bottom
            
            # Toleransi: gap harus < 50% rata-rata tinggi
            if vertical_gap > avg_h * 0.5:
                continue
            
            # --- Cek perbedaan ukuran font ---
            # Jangan gabungkan judul (font besar) dengan subjudul (font kecil)
            min_fs = min(item_i['font_size'], item_j['font_size'])
            max_fs = max(item_i['font_size'], item_j['font_size'])
            if max_fs > min_fs * 1.5:
                continue
            
            # --- Cek overlap horizontal ---
            i_left, i_right = item_i['x'], item_i['x'] + item_i['w']
            j_left, j_right = item_j['x'], item_j['x'] + item_j['w']
            
            overlap = min(i_right, j_right) - max(i_left, j_left)
            min_width = min(i_right - i_left, j_right - j_left)
            
            # Overlap harus > 30% dari lebar terkecil
            if min_width > 0 and overlap > min_width * 0.3:
                union(i, j)
    
    # --- Kumpulkan grup ---
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # --- Bangun item tergabung ---
    merged = []
    for indices in groups.values():
        if len(indices) == 1:
            # Item tunggal: tetap apa adanya
            item = items[indices[0]].copy()
            item.setdefault('multiline', False)
            item.setdefault('align', 'left')
            merged.append(item)
        else:
            # Grup multiline: gabungkan
            group_items = sorted([items[i] for i in indices], key=lambda x: x['y'])
            
            # Bounding box gabungan
            x_min = min(it['x'] for it in group_items)
            y_min = min(it['y'] for it in group_items)
            x_max = max(it['x'] + it['w'] for it in group_items)
            y_max = max(it['y'] + it['h'] for it in group_items)
            
            total_w = x_max - x_min
            total_h = y_max - y_min
            num_lines = len(group_items)
            
            # --- Deteksi alignment ---
            lefts = [it['x'] for it in group_items]
            left_std = np.std(lefts)
            
            centers = [it['x'] + it['w'] / 2 for it in group_items]
            center_std = np.std(centers)
            avg_w = np.mean([it['w'] for it in group_items])
            
            # Prioritaskan left-alignment karena ini paling umum. 
            # Jika semua teks rata kiri, left_std akan sangat kecil (< 15 pixel).
            if left_std < 15 or left_std < avg_w * 0.05:
                align = 'left'
            elif center_std < 20 or center_std < avg_w * 0.1:
                align = 'center'
            else:
                align = 'left'  # Default fallback
            
            # Teks digabung dengan newline
            combined_text = '\n'.join(it['text'] for it in group_items)
            
            # Font size: median dari semua baris
            median_fs = int(np.median([it['font_size'] for it in group_items]))
            
            # --- Hitung line-height agar teks muat di bounding box ---
            # Total tinggi = num_lines * font_size * line_height
            # Maka: line_height = total_h / (num_lines * font_size)
            if median_fs > 0 and num_lines > 0:
                computed_lh = total_h / (num_lines * median_fs)
                computed_lh = round(max(1.1, min(1.5, computed_lh)), 2)
            else:
                computed_lh = 1.2
            
            # Warna: gunakan yang paling sering muncul
            colors = [it['color'] for it in group_items]
            color = Counter(colors).most_common(1)[0][0]
            
            # Weight: gunakan yang paling sering muncul
            weights = [it.get('weight', 'normal') for it in group_items]
            weight = Counter(weights).most_common(1)[0][0]
            
            merged.append({
                'text': combined_text,
                'x': x_min,
                'y': y_min,
                'w': total_w,
                'h': total_h,
                'color': color,
                'font_size': median_fs,
                'weight': weight,
                'align': align,
                'multiline': True,
                'line_height': computed_lh
            })
    
    return merged


def process_image(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Load Gambar
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Gambar tidak ditemukan di path '{image_path}'")
        return
        
    img_height, img_width = img.shape[:2]
    
    # 2. Inisialisasi EasyOCR
    # Menggunakan bahasa Inggris + Indonesia untuk akurasi lebih baik pada slide campuran
    print("Inisialisasi EasyOCR...")
    reader = easyocr.Reader(['en', 'id'])
    
    print(f"Menjalankan EasyOCR pada gambar: {image_path}...")
    result = reader.readtext(image_path)
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    text_items = []
    
    if result:
        for idx, line in enumerate(result):
            box = line[0] # Koordinat 4 titik
            text = line[1]
            
            # Perbaiki kesalahan OCR umum: tanda koma ',' sering terbaca sebagai titik koma ';'
            text = text.replace(';', ',')
            
            confidence = line[2] if len(line) > 2 else 1.0
            
            # Filter hasil OCR dengan confidence rendah (kemungkinan noise/salah baca)
            if confidence < 0.2:
                safe_text = text.encode('ascii', 'replace').decode('ascii')
                print(f"  Skip (conf {confidence:.2f}): '{safe_text}'")
                continue
            
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            w = x_max - x_min
            h = y_max - y_min
            
            if w <= 0 or h <= 0:
                continue
                
            # --- Ekstraksi Warna, Style & Masking ---
            # Tambahkan padding agar anti-aliasing/shadow di luar bounding box ikut terhapus
            # Jika bounding box terlalu ketat, mask yang di-dilate akan terpotong di tepi box!
            pad_y = int(h * 0.25)
            pad_x = int(h * 0.25)
            
            cx1, cx2 = max(0, x_min - pad_x), min(img_width, x_max + pad_x)
            cy1, cy2 = max(0, y_min - pad_y), min(img_height, y_max + pad_y)
            
            cropped_img = img[cy1:cy2, cx1:cx2]
            
            color, text_mask = get_text_color_and_mask(cropped_img)
            
            # --- Terapkan Pixel-Perfect Mask ---
            # Masukkan pixel-mask ini ke dalam global mask sesuai koordinatnya
            mask[cy1:cy2, cx1:cx2] = cv2.bitwise_or(mask[cy1:cy2, cx1:cx2], text_mask)
            
            # --- Deteksi Font Weight ---
            # Ambil mask region yang sesuai bbox asli (tanpa padding) untuk analisis stroke
            mask_y_start = y_min - cy1
            mask_y_end = y_max - cy1
            mask_x_start = x_min - cx1
            mask_x_end = x_max - cx1
            original_mask = text_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
            weight = detect_font_weight(original_mask)
            
            # Estimasi kasaran font size dari tinggi bounding box OCR
            # Diturunkan dari 0.85 ke 0.75 agar teks memiliki sedikit ruang (padding)
            # dan tidak meluber keluar dari batas desain asli.
            estimated_font_size = int(h * 0.75)
            
            # Untuk menghindari teks meluber secara horizontal jika font sangat tebal/lebar:
            # Batasi ukuran maksimum font berdasarkan lebar karakter rata-rata
            max_width_based_fs = int((w / max(1, len(text))) * 2.5)
            estimated_font_size = min(max(10, estimated_font_size), max_width_based_fs)
            
            text_items.append({
                'text': text,
                'x': x_min,
                'y': y_min,
                'w': w,
                'h': h,
                'color': color,
                'font_size': estimated_font_size,
                'weight': weight,
                'align': 'left',
                'multiline': False
            })
            safe_text = text.encode('ascii', 'replace').decode('ascii')
            print(f"Ekstrak: '{safe_text}' di ({x_min},{y_min}) | Warna: {color} | "
                  f"Ukuran: {estimated_font_size}px | Weight: {weight} | Conf: {confidence:.2f}")
    else:
        print("Tidak ada teks yang terdeteksi pada gambar.")
    
    # --- Grouping Teks Multiline ---
    # Gabungkan teks yang secara vertikal berdekatan dan horizontal overlap
    print(f"\nGrouping {len(text_items)} item teks...")
    text_items = group_text_items(text_items)
    print(f"Hasil grouping: {len(text_items)} grup teks")
    for item in text_items:
        if item.get('multiline'):
            lines = item['text'].split('\n')
            safe_text = lines[0][:40].encode('ascii', 'replace').decode('ascii')
            print(f"  [GRUP {len(lines)} baris] '{safe_text}...' | "
                  f"Align: {item['align']} | LH: {item.get('line_height', 1.2)}")
        
    # 4. Inpainting Background
    print("\nMembersihkan teks dari background (Inpainting)...")
    # Radius diperbesar dari 3 ke 7 untuk hasil yang lebih bersih
    # Menggunakan metode Navier-Stokes (NS) karena lebih halus dalam menjaga gradasi warna
    inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
    
    # 5. Simpan Background
    bg_filename = f"{base_name}_bg.jpg"
    bg_path = os.path.join(output_dir, bg_filename)
    cv2.imwrite(bg_path, inpainted_img)
    print(f"Background berhasil disimpan ke: {bg_path}")
    
    # 6. Render HTML Interaktif
    print("Membangun struktur HTML interaktif...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    template_dir = os.path.join(script_dir, 'templates')
    
    try:
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template('template.html')
        
        html_out = template.render(
            image_width=img_width,
            image_height=img_height,
            background_image_path=bg_filename,
            text_items=text_items
        )
        
        html_filename = f"{base_name}_interactive.html"
        html_path = os.path.join(output_dir, html_filename)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_out)
            
        print(f"SELESAI! Hasil HTML interaktif ada di: {html_path}")
    except Exception as e:
        print(f"Gagal merender HTML: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layout Aware Text Extraction Pipeline")
    parser.add_argument("--image", required=True, help="Path absolute atau relatif menuju file gambar (slide)")
    parser.add_argument("--output_dir", default="outputs", help="Folder tujuan penyimpanan hasil background dan HTML")
    args = parser.parse_args()
    
    process_image(args.image, args.output_dir)

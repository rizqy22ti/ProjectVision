import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Konfigurasi folder penyimpanan sementara untuk gambar yang di-upload
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------------------------------------------------------------------
# SETUP VARIABEL & PREPROCESSING (Wajib sama persis dengan setelan di Colab)
# ------------------------------------------------------------------------------
IMG_SIZE = 128

# Daftar kelas penyakit daun Anda (urutkan persis seperti urutan alfabet folder Kaggle)
# DAFTAR KELAS ASLI 100% SINKRON DENGAN TERMINAL MODEL ANDA
CLASSES = [
    "Pepper_bell___Bacterial_spot",                  # Indeks 0
    "Pepper_bell___healthy",                         # Indeks 1
    "Potato___Early_blight",                         # Indeks 2
    "Potato___Late_blight",                          # Indeks 3
    "Potato___healthy",                              # Indeks 4
    "Tomato___Late_blight",                          # Indeks 5 (Kalibrasi: Indeks 5 adalah Late Blight)
    "Tomato___Early_blight",                         # Indeks 6
    "Tomato___Bacterial_spot",                       # Indeks 7 (Kalibrasi: Indeks 7 adalah Bacterial Spot)
    "Tomato___Leaf_Mold",                            # Indeks 8
    "Tomato___Septoria_leaf_spot",                    # Indeks 9
    "Tomato___Spider_mites_Two-spotted_spider_mite", # Indeks 10
    "Tomato___Target_Spot",                          # Indeks 11
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"         # Indeks 12
]

def build_gabor_kernels():
    """Membuat bank Gabor Filter dengan beberapa orientasi dan skala"""
    kernels = []
    ksize = 31
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]: 
        for sigma in [3.0, 5.0]:                   
            lamda = np.pi/4
            gamma = 0.5
            g_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
            kernels.append(g_kernel)
    return kernels

def apply_gabor_filter(img, kernels):
    """Menerapkan Gabor Filter dengan Masking Hijau Ketat (Sama Persis dengan Training Colab Awal)"""
    # 1. Konversi ke HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # KUNCI PENYELARASAN: Gunakan range hijau ketat agar bercak cokelat dianggap background (terpotong)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    plant_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Isolasi area daun (area bercak cokelat otomatis terpotong menjadi hitam di tahap ini)
    segmented_leaf = cv2.bitwise_and(img, img, mask=plant_mask)
    
    # 2. Jalankan Gabor Filter pada gambar grayscale daun yang sudah bolong
    gray = cv2.cvtColor(segmented_leaf, cv2.COLOR_BGR2GRAY)
    accum = np.zeros_like(gray, dtype=np.float32)
    for kernel in kernels:
        fimg = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        np.maximum(accum, fimg, accum)
        
    gabor_output = np.uint8(np.clip(accum, 0, 255))
    
    # 3. Warnai guratan Gabor dengan Jet Color Map
    gabor_colored = cv2.applyColorMap(gabor_output, cv2.COLORMAP_JET)
    gabor_leaf_only = cv2.bitwise_and(gabor_colored, gabor_colored, mask=plant_mask)
    
    # Gabungkan daun asli dengan visualisasi Gabor berwarna
    combined = cv2.addWeighted(segmented_leaf, 0.7, gabor_leaf_only, 0.3, 0)
    
    # 4. Beri warna Indigo Gelap pada background murni & bagian bercak yang terpotong
    bg_mask = cv2.bitwise_not(plant_mask)
    bg_color = np.zeros_like(img)
    bg_color[:] = [50, 15, 40] # Format BGR (Ungu Indigo Gelap)
    bg_colored = cv2.bitwise_and(bg_color, bg_color, mask=bg_mask)
    
    return cv2.add(combined, bg_colored)

# Load Model .h5 ke Memori Server
GABOR_KERNELS = build_gabor_kernels()
MODEL_PATH = 'model_penyakit_daun_gabor_cnn.h5'

print("--> Memuat Model CNN... Harap tunggu.")
model = load_model(MODEL_PATH)
print("--> Model Sukses Dimuat!")

# ------------------------------------------------------------------------------
# FLASK ROUTING (RUTE WEB)
# ------------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    # Menampilkan halaman utama web
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400
        
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        img_original = cv2.imread(filepath)
        if img_original is None:
            return jsonify({'error': 'Gagal membaca gambar'}), 400
            
        processed_gabor_large = apply_gabor_filter(img_original, GABOR_KERNELS)
        
        gabor_filename = "gabor_" + file.filename
        gabor_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gabor_filename)
        cv2.imwrite(gabor_filepath, processed_gabor_large)
        
        # 4. PREDIKSI CNN: Resize ke 128x128 khusus untuk input model CNN
        img_input_cnn = cv2.resize(processed_gabor_large, (IMG_SIZE, IMG_SIZE))
        
        # KUNCI PENYELARASAN: Ubah format BGR OpenCV ke RGB standar Keras Colab
        img_input_rgb = cv2.cvtColor(img_input_cnn, cv2.COLOR_BGR2RGB)
        
        # Lakukan normalisasi piksel menggunakan data RGB yang sudah searah dengan Colab
        input_data = np.array(img_input_rgb, dtype="float32") / 255.0
        input_data = np.expand_dims(input_data, axis=0) # Ubah ke bentuk tensor (1, 128, 128, 3)
        
        # 5. Prediksi Kelas Penyakit menggunakan model .h5
        predictions = model.predict(input_data)
        
        # ======================================================================
        # KODE DIAGNOSTIK UTAMA: Cetak nilai asli prediksi model ke terminal VS Code
        # ======================================================================
        print("\n=== HASIL DIAGNOSIS PENUH DARI MODEL H5 ===")
        for i, prob in enumerate(predictions[0]):
            nama_kelas_sementara = CLASSES[i] if i < len(CLASSES) else f"Kelas_Indeks_{i}"
            print(f"Indeks [{i}] ({nama_kelas_sementara}): {prob*100:.4f}%")
        print("===========================================\n")
        # ======================================================================
        
        # Ambil indeks dengan nilai probabilitas tertinggi
        class_idx = np.argmax(predictions)
        confidence = float(predictions[0][class_idx]) * 100
        
        # 5. Prediksi Kelas Penyakit menggunakan model .h5
        predictions = model.predict(input_data)
        
        # Ambil indeks dengan nilai probabilitas tertinggi secara otomatis
        class_idx = np.argmax(predictions)
        confidence = float(predictions[0][class_idx]) * 100
        
        # Mengambil nama dari list CLASSES yang sudah kita urutkan dengan benar
        result_label = CLASSES[class_idx].replace("___", " - ").replace("_", " ")
        
        # Mengembalikan respon data berupa format JSON ke halaman web frontend
        return jsonify({
            'class_name': result_label,
            'confidence': f"{confidence:.2f}%",
            'original_img': '/' + filepath,
            'gabor_img': '/' + gabor_filepath
        })

# WAJIB DI TEPI KIRI: Bagian ini memicu Flask memunculkan link localhost
if __name__ == '__main__':
    app.run(debug=True, port=5000)
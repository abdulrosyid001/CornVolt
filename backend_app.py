from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import io
import base64
import logging
import google.generativeai as genai
import os
from typing import Dict, List, Optional
import json

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    logger.warning("GEMINI_API_KEY not found. Chat features will be limited.")
    gemini_model = None

# Kelas penyakit
DISEASE_CLASSES = [
    'Hawar Daun (Blight)',
    'Karat Jagung (Rust)',
    'Bercak Daun Jagung (Cercospora)',
    'Sehat' 
]

# Definisi model autoencoder untuk deteksi anomali
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Fungsi untuk membuat model DenseNet
def create_densenet_model(num_classes=4):
    """Create DenseNet model with custom classifier"""
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def create_mobilenet_model(num_classes=4):
    """Create MobileNet model"""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def create_efficientnet_model(num_classes=4):
    """Create EfficientNet model"""
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def create_resnet_model(num_classes=4):
    """Create ResNet model"""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class CornDiseaseClassifier(nn.Module):
    """Custom PyTorch model untuk deteksi penyakit"""
    def __init__(self, num_classes=4):
        super(CornDiseaseClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Global variables
disease_model = None
anomaly_model = None
model_type = None
ANOMALY_THRESHOLD = 0.004719  # Ganti dengan threshold dari pelatihan

ANOMALY_TYPES = [
    'Perubahan Warna Daun',
    'Pola Bercak Tidak Normal',
    'Deformasi Struktur Daun',
    'Tekstur Permukaan Abnormal'
]

def inspect_model_file(model_path: str) -> Optional[dict]:
    """Inspect the saved model to understand its structure"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        logger.info("Model file contents:")
        if isinstance(checkpoint, dict):
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
            if 'state_dict' in checkpoint:
                logger.info("Found 'state_dict' in checkpoint")
                return checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                logger.info("Found 'model_state_dict' in checkpoint")
                return checkpoint['model_state_dict']
            else:
                logger.info("Using checkpoint as state_dict")
                return checkpoint
        else:
            logger.info("Model file contains direct state dict")
            return checkpoint
    except Exception as e:
        logger.error(f"Error inspecting model file: {e}")
        return None

def load_models() -> bool:
    """Load PyTorch models for disease and anomaly detection"""
    global disease_model, anomaly_model, model_type
    
    try:
        # Define device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Load disease model (DenseNet)
        model_path = 'best_mobilenet_model.pth'
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return False

        state_dict = inspect_model_file(model_path)
        if state_dict is None:
            logger.error("Failed to load state dictionary")
            return False

        sample_keys = list(state_dict.keys())
        logger.info(f"Sample state dict keys: {sample_keys[:10]}")

        model_types = {
            'densenet': (create_densenet_model, lambda key: 'features.denseblock' in key),
            'mobilenet': (create_mobilenet_model, lambda key: 'features.0.0.weight' in key and 'classifier.6' in key),
            'efficientnet': (create_efficientnet_model, lambda key: 'features.0.0.weight' in key and 'classifier.1' in key),
            'resnet': (create_resnet_model, lambda key: 'layer1' in key and 'fc.weight' in key),
            'custom': (CornDiseaseClassifier, lambda key: True)
        }

        selected_model_type = 'custom'
        selected_model_fn = CornDiseaseClassifier

        for m_type, (create_fn, check_fn) in model_types.items():
            if any(check_fn(key) for key in sample_keys):
                selected_model_type = m_type
                selected_model_fn = create_fn
                break

        logger.info(f"Selected model type: {selected_model_type}")

        disease_model = selected_model_fn(num_classes=len(DISEASE_CLASSES))
        model_type = selected_model_type

        try:
            disease_model.load_state_dict(state_dict)
            logger.info("State dictionary loaded successfully")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}")
            logger.info("Attempting to load with strict=False")
            disease_model.load_state_dict(state_dict, strict=False)
            logger.info("State dictionary loaded with strict=False")

        disease_model.to(device)
        disease_model.eval()
        logger.info(f"PyTorch disease model ({model_type}) loaded successfully on {device}")

        # Load anomaly model (Autoencoder)
        logger.info("Loading PyTorch anomaly detection model...")
        anomaly_model_path = "best_autoencoder.pth"
        
        if not os.path.exists(anomaly_model_path):
            logger.error(f"PyTorch anomaly model file not found at: {anomaly_model_path}")
            anomaly_model = None
        else:
            try:
                anomaly_model = Autoencoder().to(device)
                anomaly_model.load_state_dict(torch.load(anomaly_model_path, map_location=device, weights_only=True))
                anomaly_model.eval()
                logger.info("PyTorch anomaly model loaded successfully")
            except Exception as e:
                logger.error(f"PyTorch anomaly model loading failed: {e}", exc_info=True)
                logger.info("Continuing without anomaly detection model...")
                anomaly_model = None

        return True

    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        return False

def preprocess_image_pytorch(image, target_size=(224, 224)):
    """Preprocess image untuk PyTorch disease model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    processed_image = transform(image).unsqueeze(0)
    
    logger.info(f"Preprocessed disease image shape: {processed_image.shape}")
    logger.info(f"Preprocessed disease image min/max: {processed_image.min():.3f}/{processed_image.max():.3f}")
    
    return processed_image

def preprocess_image_anomaly(image, target_size=(128, 128)):
    """Preprocess image untuk PyTorch anomaly model"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()  # Normalizes to [0, 1]
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    processed_image = transform(image).unsqueeze(0)
    
    logger.info(f"Preprocessed anomaly image shape: {processed_image.shape}")
    logger.info(f"Preprocessed anomaly image min/max: {processed_image.min():.3f}/{processed_image.max():.3f}")
    
    return processed_image

def is_corn_plant_image(image):
    """Validasi apakah gambar adalah tanaman jagung"""
    try:
        disease_results = predict_disease_raw(image)
        max_confidence = max([r['confidence'] for r in disease_results])
        confidence_spread = max_confidence - min([r['confidence'] for r in disease_results])
        
        plant_related_confidence = sum([
            r['confidence'] for r in disease_results 
            if r['name'] in ['Sehat', 'Bercak Daun Jagung (Cercospora)', 
                           'Hawar Daun (Blight)', 'Karat Jagung (Rust)']
        ])
        
        anomaly_score = 0
        if anomaly_model is not None:
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                input_tensor = preprocess_image_anomaly(image).to(device)
                anomaly_model.eval()
                with torch.no_grad():
                    reconstructed = anomaly_model(input_tensor)
                    mse = torch.mean((input_tensor - reconstructed) ** 2).item()
                anomaly_score = mse
                logger.info(f"Anomaly score (MSE): {anomaly_score}")
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")
                anomaly_score = 0
        
        is_plant = True
        reasons = []
        
        if max_confidence < 60:
            is_plant = False
            reasons.append(f"Low maximum confidence: {max_confidence:.1f}%")
        
        if confidence_spread < 50:
            is_plant = False
            reasons.append(f"Uniform confidence distribution (spread: {confidence_spread:.1f}%)")
        
        if anomaly_score > ANOMALY_THRESHOLD:
            is_plant = False
            reasons.append(f"High anomaly score: {anomaly_score:.4f}")
        
        if plant_related_confidence < 50:
            is_plant = False
            reasons.append(f"Low plant-related confidence: {plant_related_confidence:.1f}%")
        
        logger.info(f"Plant validation result: {is_plant}")
        logger.info(f"Validation reasons: {reasons}")
        logger.info(f"Max confidence: {max_confidence:.1f}%, Spread: {confidence_spread:.1f}%")
        logger.info(f"Anomaly score: {anomaly_score:.4f}")
        
        return is_plant, reasons, {
            'max_confidence': max_confidence,
            'confidence_spread': confidence_spread,
            'anomaly_score': anomaly_score,
            'plant_related_confidence': plant_related_confidence
        }
        
    except Exception as e:
        logger.error(f"Error in plant validation: {e}")
        return True, ["Validation failed, assuming plant"], {}

def predict_disease_raw(image):
    """Prediksi penyakit tanpa filtering"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = preprocess_image_pytorch(image)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            outputs = disease_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
        
        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                'name': DISEASE_CLASSES[i],
                'confidence': float(prob * 100),
                'severity': get_severity_level(prob),
                'probability': float(prob)
            })
        
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results
        
    except Exception as e:
        logger.error(f"Error in raw disease prediction: {str(e)}")
        raise

def predict_disease_with_validation(image):
    """Prediksi penyakit dengan validasi gambar tanaman"""
    is_plant, reasons, metrics = is_corn_plant_image(image)
    
    if not is_plant:
        return {
            'is_plant': False,
            'reasons': reasons,
            'metrics': metrics,
            'message': 'Gambar yang diunggah sepertinya bukan tanaman jagung. Silakan upload gambar daun atau tanaman jagung.',
            'disease_results': []
        }
    
    disease_results = predict_disease_raw(image)
    
    return {
        'is_plant': True,
        'reasons': [],
        'metrics': metrics,
        'message': 'Gambar berhasil dianalisis',
        'disease_results': disease_results
    }

def predict_anomaly(image):
    """Prediksi anomali menggunakan PyTorch model"""
    if anomaly_model is None:
        return []
        
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = preprocess_image_anomaly(image).to(device)
        
        anomaly_model.eval()
        with torch.no_grad():
            reconstructed = anomaly_model(input_tensor)
            mse = torch.mean((input_tensor - reconstructed) ** 2).item()
        
        results = []
        if mse > ANOMALY_THRESHOLD:
            results.append({
                'type': 'Anomali Terdeteksi',
                'confidence': float(mse * 1000)  # Skala untuk representasi
            })
        
        logger.info(f"Anomaly prediction - MSE: {mse:.6f}, Threshold: {ANOMALY_THRESHOLD:.6f}")
        return results
        
    except Exception as e:
        logger.error(f"Error in anomaly prediction: {str(e)}")
        return []

def get_severity_level(confidence):
    """Determine severity level based on confidence"""
    if confidence > 0.8:
        return 'Parah'
    elif confidence > 0.5:
        return 'Sedang'
    elif confidence > 0.2:
        return 'Ringan'
    else:
        return 'Normal'

def generate_gemini_suggestions(disease_results: List[Dict], anomaly_results: List[Dict]) -> Optional[Dict]:
    if not gemini_model:
        logger.warning("Gemini model not initialized, using fallback suggestions")
        return generate_fallback_suggestions(disease_results[0] if disease_results else None)
    
    try:
        context = "Anda adalah seorang ahli pertanian yang berspesialisasi dalam penyakit tanaman jagung. "
        context += "Berdasarkan hasil deteksi penyakit menggunakan machine learning, berikan saran yang komprehensif.\n\n"
        
        context += "Hasil Deteksi Penyakit:\n"
        for i, result in enumerate(disease_results[:3]):
            context += f"{i+1}. {result['name']}: {result['confidence']:.1f}% (Tingkat: {result['severity']})\n"
        
        if anomaly_results:
            context += "\nAnomalitas yang Terdeteksi:\n"
            for anomaly in anomaly_results:
                context += f"- {anomaly['type']}: {anomaly['confidence']:.1f}%\n"
        
        prompt = context + """
        Berdasarkan data di atas, berikan respons HANYA dalam format JSON murni dengan struktur berikut:
        {
            "diagnosis": "Diagnosis lengkap kondisi tanaman",
            "confidence_analysis": "Analisis tingkat kepercayaan hasil deteksi",
            "treatment_recommendations": [],
            "prevention_tips": [],
            "severity_assessment": "Penilaian tingkat keparahan",
            "immediate_actions": [],
            "long_term_care": [],
            "warning_signs": []
        }
        Pastikan respons dalam bahasa Indonesia yang mudah dipahami petani. Fokus pada solusi praktis. Jangan tambahkan teks di luar JSON.
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Coba ekstrak JSON dari respons
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        try:
            suggestions = json.loads(response_text)
            logger.info("Gemini suggestions generated successfully")
            return suggestions
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response: {e}, raw response: {response_text}")
            return {
                "diagnosis": "Gagal memproses respons AI",
                "raw_response": response_text,
                "confidence_analysis": f"Berdasarkan deteksi model dengan confidence {disease_results[0]['confidence']:.1f}%",
                "treatment_recommendations": ["Konsultasi lebih lanjut diperlukan"],
                "prevention_tips": ["Ikuti saran dari ahli AI"],
                "severity_assessment": disease_results[0]['severity'] if disease_results else "Normal",
                "immediate_actions": ["Konsultasi lebih lanjut diperlukan"],
                "long_term_care": ["Pemantauan berkala"],
                "warning_signs": ["Perhatikan perubahan pada tanaman"]
            }
            
    except Exception as e:
        logger.error(f"Error generating Gemini suggestions: {str(e)}", exc_info=True)
        return generate_fallback_suggestions(disease_results[0] if disease_results else None)

def generate_fallback_suggestions(top_disease):
    """Generate fallback suggestions when Gemini is not available"""
    if not top_disease:
        return None
    
    suggestions_db = {
        'Sehat': {
            'diagnosis': 'Tanaman jagung dalam kondisi sehat berdasarkan analisis AI.',
            'confidence_analysis': f'Tingkat keyakinan {top_disease["confidence"]:.1f}% menunjukkan kondisi tanaman sangat baik.',
            'treatment_recommendations': [
                'Lanjutkan perawatan rutin yang sudah berjalan',
                'Pantau kondisi tanaman secara berkala',
                'Pastikan nutrisi dan air tercukupi'
            ],
            'prevention_tips': [
                'Pertahankan praktik budidaya yang baik',
                'Lakukan monitoring rutin mingguan',
                'Jaga kebersihan area tanam'
            ],
            'severity_assessment': 'Normal',
            'immediate_actions': [
                'Tidak ada tindakan segera yang diperlukan'
            ],
            'long_term_care': [
                'Pemeliharaan rutin sesuai jadwal',
                'Evaluasi berkala kondisi tanaman'
            ],
            'warning_signs': [
                'Perhatikan perubahan warna daun',
                'Waspadai munculnya bercak atau noda'
            ]
        },
        'Bercak Daun Jagung': {
            'diagnosis': 'Tanaman terinfeksi penyakit bercak daun (Cercospora) berdasarkan analisis AI.',
            'confidence_analysis': f'Tingkat keyakinan {top_disease["confidence"]:.1f}% mengindikasikan perlunya tindakan segera.',
            'treatment_recommendations': [
                'Aplikasi fungisida mankozeb atau klorotalonil',
                'Buang dan musnahkan daun yang terinfeksi',
                'Semprot dengan interval 7-10 hari'
            ],
            'prevention_tips': [
                'Hindari penyiraman pada daun',
                'Tingkatkan sirkulasi udara',
                'Lakukan rotasi tanaman'
            ],
            'severity_assessment': top_disease['severity'],
            'immediate_actions': [
                'Isolasi tanaman yang terinfeksi',
                'Aplikasi fungisida segera'
            ],
            'long_term_care': [
                'Perbaiki drainase area tanam',
                'Monitor perkembangan penyakit'
            ],
            'warning_signs': [
                'Penyebaran bercak ke daun lain',
                'Perubahan warna daun menjadi kuning'
            ]
        },
        'Hawar Daun': {
            'diagnosis': 'Terdeteksi infeksi hawar daun yang dapat mengurangi hasil panen.',
            'confidence_analysis': f'Dengan tingkat keyakinan {top_disease["confidence"]:.1f}%, diperlukan perhatian khusus.',
            'treatment_recommendations': [
                'Semprot fungisida sistemik',
                'Pangkas bagian tanaman yang terinfeksi',
                'Atur jarak tanam untuk sirkulasi udara'
            ],
            'prevention_tips': [
                'Gunakan varietas tahan penyakit',
                'Hindari kelembaban berlebih',
                'Sanitasi alat pertanian'
            ],
            'severity_assessment': top_disease['severity'],
            'immediate_actions': [
                'Pemangkasan bagian terinfeksi',
                'Aplikasi fungisida'
            ],
            'long_term_care': [
                'Perbaikan sistem drainase',
                'Monitoring intensif'
            ],
            'warning_signs': [
                'Penyebaran ke batang utama',
                'Layu pada daun muda'
            ]
        },
        'Karat Jagung': {
            'diagnosis': 'Infeksi jamur karat terdeteksi dengan bintik-bintik kecoklatan pada daun.',
            'confidence_analysis': f'Tingkat keyakinan {top_disease["confidence"]:.1f}% menunjukkan perlunya tindakan pengendalian.',
            'treatment_recommendations': [
                'Aplikasi fungisida triazol',
                'Tingkatkan drainase area tanam',
                'Kurangi kelembaban di sekitar tanaman'
            ],
            'prevention_tips': [
                'Tanam varietas tahan karat',
                'Hindari penanaman terlalu rapat',
                'Bersihkan gulma di sekitar tanaman'
            ],
            'severity_assessment': top_disease['severity'],
            'immediate_actions': [
                'Aplikasi fungisida segera',
                'Perbaikan drainase'
            ],
            'long_term_care': [
                'Monitoring rutin',
                'Pengelolaan lingkungan mikro'
            ],
            'warning_signs': [
                'Peningkatan jumlah pustul karat',
                'Daun mulai menguning'
            ]
        }
    }
    
    for disease_name, suggestion in suggestions_db.items():
        if disease_name.lower() in top_disease['name'].lower():
            return suggestion
    
    return suggestions_db['Sehat']

def generate_gemini_chat_response(message: str, context: str = "") -> str:
    if not gemini_model:
        logger.warning("Gemini model not initialized, using fallback response")
        return generate_fallback_chat_response(message)
    
    try:
        system_prompt = """
        Anda adalah AI assistant ahli pertanian yang berspesialisasi dalam tanaman jagung dan penyakit tanaman. 
        Berikan jawaban yang akurat, praktis, dan dalam bahasa Indonesia yang mudah dipahami.
        Jangan gunakan Markdown atau backtick dalam respons.
        """
        
        full_prompt = system_prompt + f"\n\nKonteks tambahan: {context}\n\nPertanyaan pengguna: {message}"
        
        response = gemini_model.generate_content(full_prompt)
        response_text = response.text.strip()
        logger.info("Gemini chat response generated successfully")
        return response_text
        
    except Exception as e:
        logger.error(f"Error generating Gemini chat response: {str(e)}", exc_info=True)
        return generate_fallback_chat_response(message)

def generate_fallback_chat_response(message: str) -> str:
    """Generate fallback chat response when Gemini is not available"""
    responses = {
        'penyakit': "Untuk mengatasi penyakit tanaman jagung, identifikasi terlebih dahulu jenis penyakitnya. Upload foto tanaman Anda untuk analisis otomatis, atau konsultasikan dengan ahli pertanian setempat.",
        'pupuk': "Tanaman jagung membutuhkan pupuk N-P-K dengan rasio seimbang. Berikan pupuk dasar saat tanam dan pupuk susulan 2-3 kali selama masa pertumbuhan. Sesuaikan dosis dengan kondisi tanah.",
        'hama': "Hama utama jagung termasuk ulat grayak, penggerek batang, dan wereng. Lakukan monitoring rutin dan aplikasi pestisida sesuai ambang ekonomi. Gunakan perangkap feromon untuk monitoring.",
        'tanam': "Waktu tanam jagung optimal adalah awal musim hujan. Jarak tanam 75x25 cm atau 70x20 cm. Pilih varietas unggul dan benih bersertifikat untuk hasil maksimal.",
        'air': "Tanaman jagung membutuhkan air 400-600mm selama satu musim tanam. Kekurangan air pada fase berbunga sangat kritis. Atur sistem irigasi yang baik."
    }
    
    message_lower = message.lower()
    for keyword, response in responses.items():
        if keyword in message_lower:
            return response
    
    return "Terima kasih atas pertanyaan Anda tentang pertanian jagung. Untuk bantuan lebih spesifik, silakan upload foto tanaman atau ajukan pertanyaan yang lebih detail mengenai budidaya, penyakit, atau perawatan tanaman jagung."

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'disease_model_loaded': disease_model is not None,
        'anomaly_model_loaded': anomaly_model is not None,
        'gemini_available': gemini_model is not None,
        'model_type': model_type
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        image = Image.open(file.stream)
        logger.info(f"Image loaded: {image.size}, mode: {image.mode}")
        
        disease_result = predict_disease_with_validation(image)
        
        if not disease_result['is_plant']:
            return jsonify({
                'success': False,
                'message': disease_result['message'],
                'validation_failed': True,
                'reasons': disease_result['reasons'],
                'metrics': disease_result['metrics']
            })
        
        anomaly_results = predict_anomaly(image)
        
        ai_suggestions = generate_gemini_suggestions(
            disease_result['disease_results'], 
            anomaly_results
        )
        
        response = {
            'success': True,
            'message': 'Analisis berhasil dilakukan',
            'disease_detection': {
                'results': disease_result['disease_results'],
                'top_prediction': disease_result['disease_results'][0] if disease_result['disease_results'] else None
            },
            'anomaly_detection': {
                'results': anomaly_results,
                'detected': len(anomaly_results) > 0
            },
            'ai_suggestions': ai_suggestions,
            'validation_metrics': disease_result['metrics'],
            'model_info': {
                'disease_model_type': model_type,
                'anomaly_model_available': anomaly_model is not None,
                'ai_suggestions_source': 'gemini' if gemini_model else 'fallback'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Terjadi kesalahan saat memproses gambar'
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint for agricultural consultation"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        message = data['message']
        context = data.get('context', '')
        
        response = generate_gemini_chat_response(message, context)
        
        return jsonify({
            'success': True,
            'response': response,
            'ai_source': 'gemini' if gemini_model else 'fallback'
        })
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Terjadi kesalahan saat memproses pertanyaan'
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'disease_model': {
            'loaded': disease_model is not None,
            'type': model_type,
            'classes': DISEASE_CLASSES
        },
        'anomaly_model': {
            'loaded': anomaly_model is not None,
            'types': ANOMALY_TYPES
        },
        'ai_features': {
            'gemini_available': gemini_model is not None,
            'chat_enabled': True,
            'suggestions_enabled': True
        }
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Corn Disease Detection API',
        'version': '2.0',
        'endpoints': {
            'health': '/api/health',
            'predict': '/api/predict (POST)',
            'chat': '/api/chat (POST)',
            'model_info': '/api/model-info'
        }
    })

if __name__ == '__main__':
    logger.info("Starting Corn Disease Detection API...")
    if load_models():
        logger.info("Models loaded successfully. Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load models. Exiting...")

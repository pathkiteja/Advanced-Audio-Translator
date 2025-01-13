from flask import Flask, render_template, request, send_file, redirect, url_for
import torch
import torchaudio
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from gtts import gTTS
import os
import webbrowser
import threading

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load Whisper Model
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model.to(device)

def transcribe_audio(audio_path, target_language='en'):
    speech_array, sampling_rate = sf.read(audio_path, dtype='float32')
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_tensor = torch.tensor(speech_array).unsqueeze(0)
        speech_tensor = resampler(speech_tensor).squeeze(0)
        speech_array = speech_tensor.numpy()

    inputs = whisper_processor(speech_array, sampling_rate=16000, return_tensors="pt").to(device)
    predicted_ids = whisper_model.generate(inputs["input_features"], task="translate", language=target_language)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_path = os.path.join(RESULT_FOLDER, "text_input_audio.mp3")
    tts.save(audio_path)
    return audio_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio_file' not in request.files:
        return redirect(url_for('index'))

    audio_file = request.files['audio_file']
    language = request.form.get('language', 'en')

    if audio_file.filename == '':
        return redirect(url_for('index'))

    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)

    transcription = transcribe_audio(file_path, language)

    result_path = os.path.join(RESULT_FOLDER, "transcription.txt")
    with open(result_path, 'w') as f:
        f.write(transcription)

    # Generate TTS audio
    text_to_speech(transcription, language)

    return render_template('index.html', transcription=transcription)

@app.route('/submit_text', methods=['POST'])
def submit_text():
    user_text = request.form.get('user_text')
    language = request.form.get('text_language', 'en')

    # Generate audio from user text
    audio_path = text_to_speech(user_text, language)

    return render_template('index.html', tts_text=user_text)

@app.route('/download_transcription')
def download_transcription():
    result_path = os.path.join(RESULT_FOLDER, "transcription.txt")
    return send_file(result_path, as_attachment=True)

@app.route('/play_text_audio')
def play_text_audio():
    audio_path = os.path.join(RESULT_FOLDER, "text_input_audio.mp3")
    return send_file(audio_path, as_attachment=False)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True)

import whisper
from pyannote.audio import Pipeline
import os
import pandas as pd
import openpyxl
import ctypes

libc_name = "msvcrt.dll"
libc = ctypes.CDLL(libc_name)

# Whisper modeli yükle
whisper_model = whisper.load_model("large")

# Pyannote pipeline yükle (önceden kimlik doğrulama gerektirir)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")

# Aranacak kelimeler listesi
keywords = ["küfür0", "küfür1", "küfür2"]

# Ses dosyalarının bulunduğu dizin
sound_files_dir = "speech"

# Tüm wav dosyalarını bul
wav_files = [f for f in os.listdir(sound_files_dir) if f.endswith('.wav')]

for audio_file in wav_files:
    audio_path = os.path.join(sound_files_dir, audio_file)

    # Ses dosyasını yükle ve transkripte et
    result = whisper_model.transcribe(audio_path)
    transcript = result["text"]

    # Pyannote ile konuşmacı diarizasyonu (MP3 yerine WAV dosyasını kullan)
    diarization = pipeline(audio_path)

    # Konuşmacı ID'lerini zaman damgalarıyla eşleştirin
    speaker_turns = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns[(turn.start, turn.end)] = speaker

    # Konuşmacı ID'leri ile metni birleştirin
    dialogue = []
    current_speaker = None
    current_segment = []
    for segment in result["segments"]:
        start, end = segment["start"], segment["end"]
        text = segment["text"]

        # En yakın konuşmacı segmentini bulun
        speaker = None
        for (s_start, s_end), s_id in speaker_turns.items():
            if s_start <= start <= s_end or s_start <= end <= s_end:
                speaker = s_id
                break

        if speaker != current_speaker:
            if current_segment:
                dialogue.append((current_speaker, " ".join(current_segment)))
                current_segment = []
            current_speaker = speaker

        current_segment.append(text)

    # Son segmenti ekle
    if current_segment:
        dialogue.append((current_speaker, " ".join(current_segment)))

    # Sonuçları ve kelime aramasını kontrol et
    alert = False
    rows = []
    for speaker, text in dialogue:
        for keyword in keywords:
            if keyword in text:
                alert = True
                break
        rows.append([speaker, text])

    # Uyarı varsa dosya adının başına "Terbiyesiz Herif..!" ekle
    if alert:
        output_filename = f"Terbiyesiz Herif..! {audio_file.split('.')[0]}.xlsx"
    else:
        output_filename = f"{audio_file.split('.')[0]}.xlsx"
    
    output_path = os.path.join(sound_files_dir, output_filename)

    # Sonuçları Excel dosyasına yazdır
    df = pd.DataFrame(rows, columns=["Speaker", "Text"])
    df.to_excel(output_path, index=False)

    print(f"Dosya işlendi: {output_path}")
 
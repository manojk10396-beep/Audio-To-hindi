
import os
import asyncio
import uuid
import edge_tts
from flask import Flask, request, render_template_string, send_from_directory
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

whisper = WhisperModel("base", device="cpu", compute_type="int8")

translator = None
tokenizer = None

HTML = """
<h1>Audio to Hindi AI</h1>
<h3>किसी भी ऑडियो को हिंदी में बदलें</h3>
<form method="POST" enctype="multipart/form-data">
<input type="file" name="audio" accept="audio/*,video/*" required>
<button type="submit">हिंदी में बदलें</button>
</form>
{% if text %}
<h3>हिंदी में अनुवाद:</h3>
<p>{{ text }}</p>
<a href="/download/{{ audio }}">हिंदी आवाज डाउनलोड करें</a>
{% endif %}
{% if error %}<p>{{ error }}</p>{% endif %}
"""

LANGUAGES = {
    "en": "eng_Latn", "hi": "hin_Deva",
    "fr": "fra_Latn", "de": "deu_Latn",
    "es": "spa_Latn", "ar": "arb_Arab",
    "zh": "zho_Hans", "ja": "jpn_Jpan",
    "ru": "rus_Cyrl", "pt": "por_Latn",
    "it": "ita_Latn", "ko": "kor_Hang",
    "tr": "tur_Latn", "nl": "nld_Latn"
}

def translate(text, lang):
    global tokenizer, translator

    if translator is None:
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        translator = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    source = LANGUAGES.get(lang, "eng_Latn")
    tokenizer.src_lang = source
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=400)

    result = translator.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("hin_Deva"),
        max_length=400
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)[0]

async def make_voice(text, filename):
    voice = edge_tts.Communicate(text, "hi-IN-MadhurNeural")
    await voice.save(filename)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("audio")
        if not file or not file.filename:
            return render_template_string(HTML, error="ऑडियो फाइल चुनें")

        uid = str(uuid.uuid4())
        path = os.path.join("uploads", uid + "_" + file.filename)
        file.save(path)

        try:
            segments, info = whisper.transcribe(path)
            original = " ".join(s.text for s in segments)

            if not original.strip():
                raise ValueError("ऑडियो में आवाज नहीं मिली")

            hindi = translate(original, info.language)
            audio_name = uid + ".mp3"
            asyncio.run(make_voice(
                hindi, os.path.join("outputs", audio_name)
            ))

            return render_template_string(
                HTML, text=hindi, audio=audio_name
            )
        except Exception as e:
            return render_template_string(HTML, error=str(e))

    return render_template_string(HTML)

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory("outputs", filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
  

import torch
import torchaudio
import numpy as np
from transformers import AutoProcessor, SeamlessM4Tv2Model
import os
import gradio as gr

# Dictionnaire des langues avec correspondance entre noms d'affichage et codes
LANGUAGES = {
    "Français": "fra",
    "Anglais": "eng",
    "Arabe standard": "arb",
    "Darija": "ary",
    "Espagnol": "spa",
    "Allemand": "deu",
    "Russe": "rus",
    "Chinois": "cmn",
    "Italien": "ita",
    "Portugais": "por",
    "Japonais": "jpn"
}

class SeamlessModel:
    def __init__(self):
        print("Début de la fonction __init__")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation de l'appareil: {self.device}")
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        self.model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(self.device)
        print("Fin de la fonction __init__")

    def speech_to_text(self, audio_array, src_lang):
        print("Début de la fonction speech_to_text")
        inputs = self.processor(audios=audio_array, return_tensors="pt", sampling_rate=16000).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                tgt_lang="eng",  # On traduit toujours en anglais d'abord
                generate_speech=False,
                num_beams=5,
                length_penalty=0.6,
            )

        if hasattr(outputs, 'sequences'):
            token_ids = outputs.sequences[0].cpu().numpy().tolist()
            transcription = self.processor.decode(token_ids, skip_special_tokens=True)
        else:
            raise AttributeError("Unexpected output format from model.generate()")
        print("Fin de la fonction speech_to_text")
        return transcription

    def text_to_speech(self, text, src_lang, tgt_lang):
        print("Début de la fonction text_to_speech")
        inputs = self.processor(text=text, src_lang=src_lang, return_tensors="pt").to(self.device)
        with torch.no_grad():
            audio_array = self.model.generate(**inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
        print("Fin de la fonction text_to_speech")
        return audio_array

    def translate_text(self, text, src_lang, tgt_lang):
        print("Début de la fonction translate_text")
        inputs = self.processor(text=text, src_lang=src_lang, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)

        if hasattr(outputs, 'sequences'):
            token_ids = outputs.sequences[0].cpu().tolist()
        elif isinstance(outputs, torch.Tensor):
            token_ids = outputs[0].cpu().tolist()
        else:
            raise ValueError(f"Unexpected output type: {type(outputs)}")

        translated_text = self.processor.decode(token_ids, skip_special_tokens=True)
        print("Fin de la fonction translate_text")
        return translated_text

def process_audio(audio_path, input_lang_code, output_lang_code, model=None):
    """
    Fonction pour traiter l'audio
    """
    print(f"Traitement de l'audio: {audio_path}")
    print(f"Langue d'entrée: {input_lang_code}")
    print(f"Langue de sortie: {output_lang_code}")

    # Créer ou réutiliser le modèle
    if model is None:
        model = SeamlessModel()

    # Gérer les différents formats d'entrée de Gradio
    if isinstance(audio_path, tuple):  # Format (sr, data)
        sample_rate, audio_data = audio_path
        # Convertir en mono si stéréo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
    else:  # Chemin de fichier
        print(f"Chargement du fichier audio: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:  # Convertir en mono si stéréo
            waveform = waveform.mean(dim=0, keepdim=True)
        audio_data = waveform.squeeze().numpy()

    # Rééchantillonner à 16kHz si nécessaire
    if sample_rate != 16000:
        print(f"Rééchantillonnage de {sample_rate}Hz à 16000Hz")
        audio_tensor = torch.tensor(audio_data).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        resampled_tensor = resampler(audio_tensor)
        audio_data = resampled_tensor.squeeze().numpy()

    # Transcrire l'audio en texte
    transcription = model.speech_to_text(audio_data, input_lang_code)
    print(f"Transcription: {transcription}")

    # Traduire le texte
    translated_text = model.translate_text(transcription, src_lang="eng", tgt_lang=output_lang_code)
    print(f"Traduction: {translated_text}")

    # Générer l'audio de sortie
    output_audio = model.text_to_speech(translated_text, src_lang=output_lang_code, tgt_lang=output_lang_code)

    # Sauvegarder l'audio temporairement pour Gradio
    output_file = f"output_audio_{output_lang_code}_{np.random.randint(1000)}.wav"
    output_sample_rate = model.model.config.sampling_rate
    torchaudio.save(output_file, torch.tensor(output_audio).unsqueeze(0), output_sample_rate)

    # Retourner les résultats pour l'interface Gradio
    return transcription, translated_text, output_file

def translate_darija_to_english(audio):
    """
    Fonction pour traduire l'audio du darija vers l'anglais
    """
    if audio is None:
        return "Aucun audio fourni", "Aucune traduction disponible", None

    try:
        return process_audio(
            audio,
            input_lang_code=LANGUAGES["Darija"],
            output_lang_code=LANGUAGES["Anglais"],
            model=seamless_model
        )
    except Exception as e:
        print(f"Erreur lors de la traduction darija-anglais: {e}")
        return f"Erreur: {str(e)}", "Échec de la traduction", None

def translate_english_to_arabic(audio):
    """
    Fonction pour traduire l'audio de l'anglais vers l'arabe standard
    """
    if audio is None:
        return "Aucun audio fourni", "Aucune traduction disponible", None

    try:
        return process_audio(
            audio,
            input_lang_code=LANGUAGES["Anglais"],
            output_lang_code=LANGUAGES["Arabe standard"],
            model=seamless_model
        )
    except Exception as e:
        print(f"Erreur lors de la traduction anglais-arabe: {e}")
        return f"Erreur: {str(e)}", "Échec de la traduction", None

# Charger le modèle une seule fois
print("Chargement du modèle Seamless M4T...")
seamless_model = SeamlessModel()
print("Modèle chargé avec succès!")

# Créer l'interface Gradio simplifiée
with gr.Blocks(title="Traducteur Darija-Anglais-Arabe") as app:
    gr.Markdown("# 🎙️ Traducteur Darija ↔ Anglais ↔ Arabe")
    gr.Markdown("Traduisez facilement entre Darija, Anglais et Arabe avec un seul clic")

    with gr.Row():
        # Colonne Darija -> Anglais
        with gr.Column():
            gr.Markdown("## Darija → Anglais")
            audio_input_darija = gr.Audio(
                label="Enregistrement en Darija",
                type="filepath",
                sources=["microphone", "upload"]
            )
            btn_darija = gr.Button("Traduire en Anglais", variant="primary")

            with gr.Row():
                text_darija = gr.Textbox(label="Texte transcrit (Darija)")

            # Ces composants seront également des sorties pour la traduction Anglais -> Arabe
            with gr.Row():
                text_arabic = gr.Textbox(label="Texte traduit (Arabe)", elem_id="arabic_text")

            audio_output_arabic = gr.Audio(label="Audio traduit (Arabe)", elem_id="arabic_audio")

        # Colonne Anglais -> Arabe
        with gr.Column():
            gr.Markdown("## Anglais → Arabe")
            audio_input_english = gr.Audio(
                label="Enregistrement en Anglais",
                type="filepath",
                sources=["microphone", "upload"]
            )
            btn_english = gr.Button("Traduire en Arabe", variant="primary")

            with gr.Row():
                text_english_src = gr.Textbox(label="Texte transcrit (Anglais)")

            # Ces composants seront également des sorties pour la traduction Darija -> Anglais
            with gr.Row():
                text_english = gr.Textbox(label="Texte traduit (Anglais)", elem_id="english_text")

            audio_output_english = gr.Audio(label="Audio traduit (Anglais)", elem_id="english_audio")

    # Configurer les événements
    btn_darija.click(
        fn=translate_darija_to_english,
        inputs=[audio_input_darija],
        outputs=[text_darija, text_english, audio_output_english]
    )

    btn_english.click(
        fn=translate_english_to_arabic,
        inputs=[audio_input_english],
        outputs=[text_english_src, text_arabic, audio_output_arabic]
    )

    gr.Markdown("""
    ## Instructions d'utilisation

    ### À gauche:
    1. Enregistrez votre voix en Darija (dialecte marocain)
    2. Cliquez sur "Traduire en Anglais"
    3. La traduction anglaise apparaîtra à droite

    ### À droite:
    1. Enregistrez votre voix en Anglais
    2. Cliquez sur "Traduire en Arabe"
    3. La traduction arabe apparaîtra à gauche

    Cette interface permet d'avoir une conversation fluide entre ces trois langues.
    """)

# Lancer l'application
if __name__ == "__main__":
    app.launch()
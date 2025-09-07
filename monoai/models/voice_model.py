from litellm import speech
from ..keys.keys_manager import load_key
import base64
import os

class VoiceModel:
    
    def __init__(self, provider: str, model: str, voice: str):
        load_key(provider)
        self._provider = provider
        self._model = model
        self._voice = voice
        
        if self._provider == "elevenlabs":
            try:
                from elevenlabs.client import ElevenLabs
            except ImportError:
                raise ImportError("elevenlabs is not installed. Please install it with 'pip install elevenlabs'")

            self._client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        
    def speak(self, text: str, min_chars_per_sentence=100, return_type="audio.mp3"):
        
        audio_groups = self._generate_audio_groups(text, min_chars_per_sentence)
        audio_chunks = []
        
        for group in audio_groups:
            
            response = self._generate(group)
            
            audio_chunks.append(response)
        
        if return_type.endswith(".mp3") or return_type.endswith(".wav"):
            combined_audio = self._combine_bytes_chunks(audio_chunks)
            with open(return_type, 'wb') as f:
                f.write(combined_audio)
            return return_type
        elif return_type == "base64":
            combined_audio = self._combine_bytes_chunks(audio_chunks)
            return base64.b64encode(combined_audio).decode('utf-8')
        elif return_type == "bytes":
            return self._combine_bytes_chunks(audio_chunks)
        else:
            raise ValueError(f"Invalid return type: {return_type}")
    
    def _combine_bytes_chunks(self, audio_chunks: list) -> bytes:
        """Unisce i chunk audio in un unico oggetto bytes"""
        combined = b""
        for chunk in audio_chunks:
            combined += chunk
        return combined
    
    def _generate_audio_groups(self, text: str, min_chars_per_sentence: int):
        """Genera gruppi di sentences ottimizzati per lunghezza"""
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        audio_groups = []
        i = 0
        
        while i < len(sentences):
            current_group = sentences[i]
            j = i + 1
            
            # Continua ad aggiungere sentences successive finché non superi min_chars_per_sentence
            while j < len(sentences):
                next_sentence = sentences[j]
                combined_length = len(current_group + ". " + next_sentence)
                
                # Se l'unione supera il limite, fermati
                if combined_length > min_chars_per_sentence:
                    break
                
                # Altrimenti, unisci la sentence successiva
                current_group += ". " + next_sentence
                j += 1
            
            audio_groups.append(current_group)
            # Passa alla prossima sentence non processata
            i = j
        
        return audio_groups
        
    def _generate(self, text):
        
        if self._provider == "elevenlabs":
            response = self._client.text_to_speech.convert(
                text=text,
                voice_id=self._voice,
                model_id=self._model,
                output_format="mp3_44100_128",
            )
            response_bytes = b""
            for r in response:
                response_bytes += r
            return response_bytes
        else:
            response = speech(
                model=self._provider+"/"+self._model,
                voice=self._voice,
                input=text,
            )
            return response.content
    
    async def stream(self, text: str, min_chars_per_sentence=100, return_type="audio.mp3"):
        
        audio_groups = self._generate_audio_groups(text, min_chars_per_sentence)
        
        for _, group in enumerate(audio_groups):
            # Genera audio per il gruppo di sentences
            response = self._generate(group)

            if return_type.endswith(".mp3") or return_type.endswith(".wav"):
                with open(return_type, 'wb') as f:
                    f.write(response)
                yield return_type
            elif return_type == "base64":
                yield base64.b64encode(response).decode('utf-8')
            elif return_type == "bytes":
                yield response
            else:
                raise ValueError(f"Invalid return type: {return_type}")

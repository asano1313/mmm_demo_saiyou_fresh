import os
import json
from deepgram import DeepgramClient, PrerecordedOptions
import streamlit as st

os.environ['DEEPGRAM_API_KEY'] = st.secrets.DEEPGRAM_API_KEY.key
#Deepgramで文字おこし
class DeepgramTranscriber:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
      raise ValueError("api_key is required. set DEEPGRAM_API_KEY")

    def __init__(self, audio_file_path, json_file_path=None, transcript_file_path=None):
        self.audio_file_path = audio_file_path
        self.json_file_path = json_file_path
        self.transcript_file_path = transcript_file_path
        self.deepgram = DeepgramClient(self.api_key)

    def __call__(self):
        if self.json_file_path is None or self.transcript_file_path is None:
            return self.transcribe_with_no_save()
        else:
            return self.transcribe_with_save()

    def transcribe(self):
        print(self.audio_file_path)
        print(os.path.isfile(self.audio_file_path))
        with open(self.audio_file_path, 'rb') as buffer_data:
            payload = {'buffer': buffer_data}
            options = PrerecordedOptions(
                punctuate=True,
                model="nova-2",
                language="ja",
                # diarize=True,
                # utterances=True,
                # smart_format = True,
            )
            print('Requesting transcript...')
            print('Your file may take up to a couple minutes to process.')
            print('While you wait, did you know that Deepgram accepts over 40 audio file formats? Even MP4s.')
            print('To learn more about customizing your transcripts check out developers.deepgram.com')
            response = self.deepgram.listen.prerecorded.v('1').transcribe_file(payload, options)
            return response

    def transcribe_with_no_save(self):
        response = self.transcribe()
        # for u in response.results.utterances:
        #   print(f"[Speaker: {u.speaker}] {u.transcript}")
        return response.results.channels[0].alternatives[0].transcript

    def transcribe_with_save(self):
        response = self.transcribe()
        self._save_json(response)
        self._save_transcript(response)
        return response.results.channels[0].alternatives[0].transcript

    def _save_json(self, response):
        with open(self.json_file_path, 'w') as outfile:
            json.dump(response.to_json(indent=4), outfile)

    def _save_transcript(self, response):
        transcript = response.results.channels[0].alternatives[0].transcript
        with open(self.transcript_file_path, 'w') as outfile:
            outfile.write(transcript)

    def read_json(self):
        with open(self.json_file_path, 'r') as infile:
            return json.load(infile)

    def read_transcript(self):
        with open(self.transcript_file_path, 'r') as infile:
            return infile.read()
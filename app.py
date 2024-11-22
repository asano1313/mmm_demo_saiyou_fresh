import streamlit as st
import os
import ffmpeg
import pathlib
import time
from threading import Thread
from queue import Queue
from deepgram import DeepgramClient, PrerecordedOptions
import os
import json
from pydub import AudioSegment
# from langchain_openai import ChatOpenAI
# from langchain_community.callbacks import get_openai_callback
import shutil
import boto3
import vertexai
import vertexai.generative_models as genai
from google.cloud import storage
import libs.prompts.fresh_meeting as fresh_meeting
from libs.audio import FFmpegM4AConverter, DeepgramTranscriber


os.environ['OPENAI_API_KEY'] = st.secrets.OPENAI_API_KEY.key
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets.BEDROCK_ACCESS_KEY_ID.key
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets.BEDROCK_SECRET_ACCESS_KEY.key

google_application_credentials = st.secrets["json_key"]["key"]
pathlib.Path("samplevertexai.json").write_text(google_application_credentials)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "samplevertexai.json"
BUCKET_NAME = 'dev-gemini-audio-bucket'

storage_client = storage.Client.from_service_account_json("samplevertexai.json")
bucket = storage_client.bucket(BUCKET_NAME)
vertexai.init(project='dev-rororo')

client = boto3.client("bedrock-runtime", region_name="ap-northeast-1")


def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_in_seconds = len(audio) / 1000.0  # ミリ秒から秒に変換
    duration_in_minuetes = duration_in_seconds / 60
    return duration_in_minuetes
        
def transcribe(m4a_path):
    transcriber = DeepgramTranscriber(m4a_path)
    transcript = transcriber()
    return transcript


st.set_page_config(
    page_title="デモ",
    layout="wide", # wideにすると横長なレイアウトに
    initial_sidebar_state="expanded"
)

st.title("新卒")
st.sidebar.markdown("# STEP 1 ")
interviewer_name = st.sidebar.text_input("担当者の名前を入力してください", placeholder="例:山田太郎")
job_seeker_name = st.sidebar.text_input("求職者の名前を入力してください", placeholder="例:山田花子")
st.sidebar.markdown("# STEP 2 ")
uploaded_file = st.sidebar.file_uploader(
    "動画or音声ファイルをアップロードしてください", type=["mp4", "m4a", "wav"]
)

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Gemini1.5")

with col2:
    st.header("G1.5F")

with col3:
    st.header("Claude3.5Sonnet")


def genearate(audio_file, model_name):
    if model_name == "claude":
        class MockResponse:
            def __init__(self, text):
                self.text = text
        def mock_streams():
            tran = transcribe(audio_file)
            prompt = fresh_meeting.generate_fresh_prompt_claude(interviewer_name=interviewer_name, job_seeker_name=job_seeker_name, tran=tran)
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
            })
            response = client.invoke_model_with_response_stream(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=body
            )
            for event in response.get('body'):
                if event:
                    chunk = json.loads(event.get('chunk').get('bytes').decode())
                    if chunk.get('type') == 'content_block_delta':
                        yield MockResponse(chunk.get('delta', {}).get('text', ''))

        return mock_streams()
    else:
        print("gemini")
        global price_text 
        blob = bucket.blob(audio_file)
        with open(audio_file, 'rb') as audio_file_path:
            blob.upload_from_file(audio_file_path)
        print("success")
        gcs_uri = f'gs://{BUCKET_NAME}/{audio_file}'
        print(gcs_uri)

        audio = genai.Part.from_uri(
            mime_type="audio/wav",
            uri = gcs_uri
        )

        model = genai.GenerativeModel(
            model_name
        )
        price_text = "1"

        prompt = fresh_meeting.generate_fresh_prompt_gemini(interviewer_name=interviewer_name, job_seeker_name=job_seeker_name)

        response = model.generate_content(
            [audio, prompt], stream=True
        )

        return response



if uploaded_file is not None:
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1]

    if file_extension != ".m4a":
        with st.spinner('音声ファイルに変換中...'):
            status_text = st.empty()
            save_dir = "uploaded_files"
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            converter = FFmpegM4AConverter()
            inputfile = file_path
            outputfile = save_dir
            m4a_path = converter(file_path, outputfile)
        # status_text.success('処理が完了しました!')  
    elif file_extension == ".wav":
        save_dir = "uploaded_files"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        m4a_path = file_path
    else :
        save_dir = "uploaded_files"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        m4a_path = file_path



    minutes_container1 = col1.empty()
    minutes_container2 = col2.empty()
    minutes_container3 = col3.empty()  

    def process_gemini(model_name, m4a_path, result_queue):
        minutes = ""
        response = genearate(m4a_path, model_name)
        for chunk in response:
            minutes += chunk.text
            result_queue.put((minutes, model_name))  # 結果をキューに送信
        minutes += price_text
        result_queue.put((minutes, model_name))  # 最終結果をキューに送信
        result_queue.put((None, model_name))  # 完了シグナル

    # メインの処理部分
    with st.spinner('Gemini モデル実行中...'):
        result_queue = Queue()
        
        thread1 = Thread(
            target=process_gemini,
            args=("gemini-1.5-pro-002", m4a_path, result_queue)
        )
        thread2 = Thread(
            target=process_gemini,
            args=("gemini-1.5-flash-002", m4a_path, result_queue)
        )
        thread3 = Thread(
            target=process_gemini,
            args=("claude", m4a_path, result_queue)
        )
        
        thread1.start()
        thread2.start()
        thread3.start()
        
        # 結果を受け取って表示する
        completed_models = set()
        while len(completed_models) < 3:  # 両方のモデルが完了するまで
            result, model_name = result_queue.get()
            if result is None:
                completed_models.add(model_name)
                continue
                
            if model_name == "gemini-1.5-pro-002":
                minutes_container1.write(result)
            elif model_name == "gemini-1.5-flash-002":
                minutes_container2.write(result)
            else:
                minutes_container3.write(result)
        
        thread1.join()
        thread2.join()
        thread3.join()


    directory = 'uploaded_files'

    # ディレクトリが存在するか確認
    if os.path.exists(directory):
        # ディレクトリ内のすべてのファイルとフォルダを削除
        shutil.rmtree(directory)
        # 空のディレクトリを再作成
        os.makedirs(directory)
        print(f"{directory} ディレクトリ内のすべてのファイルが削除されました。")
    else:
        print(f"{directory} ディレクトリが見つかりません。")



    

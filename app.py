import streamlit as st
import os
import ffmpeg
import pathlib
import time
from threading import Thread
from queue import Queue
# from deepgram import DeepgramClient, PrerecordedOptions
import os
import json
from pydub import AudioSegment
# from langchain_openai import ChatOpenAI
# from langchain_community.callbacks import get_openai_callback
import shutil

import vertexai
import vertexai.generative_models as genai
from google.cloud import storage


os.environ['DEEPGRAM_API_KEY'] = st.secrets.DEEPGRAM_API_KEY.key
os.environ['OPENAI_API_KEY'] = st.secrets.OPENAI_API_KEY.key

google_application_credentials = st.secrets["json_key"]["key"]
pathlib.Path("samplevertexai.json").write_text(google_application_credentials)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "samplevertexai.json"
BUCKET_NAME = 'dev-gemini-audio-bucket'

storage_client = storage.Client.from_service_account_json("samplevertexai.json")
bucket = storage_client.bucket(BUCKET_NAME)
vertexai.init(project='dev-rororo')



def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_in_seconds = len(audio) / 1000.0  # ミリ秒から秒に変換
    duration_in_minuetes = duration_in_seconds / 60
    return duration_in_minuetes


#コンバータークラス
class FFmpegM4AConverter:
    """
    FFmpegを使用してさまざまな形式のオーディオファイルをM4Aファイルに変換するクラス。

    使用方法:
    1. インスタンスを作成します。必要に応じて、オーディオ設定を指定できます。
       シンプルな使い方
       converter = FFmpegM4AConverter()

       詳細な使い方
       converter = FFmpegM4AConverter(sample_rate=48000, bitrate=256000, channels=2, bits_per_sample=16)

    2. convertメソッドまたは__call__メソッドを使用して、ファイルを変換します。
       シンプルな使い方
       converter("input.wav", "output_directory")  # デフォルトでnormalizeが適用されます
       converter.convert("input.wav", "output_directory")  # デフォルトでnormalizeが適用されます

       詳細な使い方
       converter("input.mp3", "output_directory", normalize=False, vbr=True, metadata={"artist": "John Doe", "title": "Example"})

    対応する入力ファイル形式:
    - オーディオ形式: .aac, .ac3, .aif, .aiff, .alac, .amr, .ape, .flac, .m4a, .mp3, .ogg, .opus, .wav, ...
    - ビデオ形式: .avi, .flv, .mkv, .mov, .mp4, .mpeg, .webm, .wmv, ...

    変換されたM4Aファイルは、指定された出力ディレクトリに保存されます。
    """

    DEFAULT_SAMPLE_RATE = 44100
    DEFAULT_BITRATE = 192000
    DEFAULT_CHANNELS = 1
    DEFAULT_BITS_PER_SAMPLE = 16
    DEFAULT_ADJUST_VOLUME = True
    DEFAULT_TARGET_VOLUME = -10

    def __init__(self, sample_rate=None, bitrate=None, channels=None, bits_per_sample=None, adjust_volume=None, target_volume=None):
        self.sample_rate = sample_rate or self.DEFAULT_SAMPLE_RATE
        self.bitrate = bitrate or self.DEFAULT_BITRATE
        self.channels = channels or self.DEFAULT_CHANNELS
        self.bits_per_sample = bits_per_sample or self.DEFAULT_BITS_PER_SAMPLE
        self.adjust_volume = adjust_volume if adjust_volume is not None else self.DEFAULT_ADJUST_VOLUME
        self.target_volume = target_volume or self.DEFAULT_TARGET_VOLUME
        self.supported_extensions = self._get_supported_extensions()

    def _get_supported_extensions(self):
        return [
            '.3g2', '.3gp', '.aac', '.ac3', '.aif', '.aiff', '.alac', '.amr', '.ape',
            '.asf', '.au', '.avi', '.caf', '.dts', '.dtshd', '.dv', '.eac3', '.flac',
            '.flv', '.m2a', '.m2ts', '.m4a', '.m4b', '.m4p', '.m4r', '.m4v', '.mka',
            '.mkv', '.mod', '.mov', '.mp1', '.mp2', '.mp3', '.mp4', '.mpa', '.mpc',
            '.mpeg', '.mpg', '.mts', '.nut', '.oga', '.ogg', '.ogm', '.ogv', '.ogx',
            '.opus', '.ra', '.ram', '.rm', '.rmvb', '.shn', '.spx', '.tak', '.tga',
            '.tta', '.vob', '.voc', '.wav', '.weba', '.webm', '.wma', '.wmv', '.wv',
            '.y4m', '.aac', '.aif', '.aiff', '.aiffc', '.flac', '.iff', '.m4a', '.m4b',
            '.m4p', '.mid', '.midi', '.mka', '.mp3', '.mpa', '.oga', '.ogg', '.opus',
            '.pls', '.ra', '.ram', '.spx', '.tta', '.voc', '.vqf', '.w64', '.wav',
            '.wma', '.xm', '.3gp', '.a64', '.ac3', '.amr', '.drc', '.dv', '.flv',
            '.gif', '.h261', '.h263', '.h264', '.hevc', '.m1v', '.m4v', '.mkv', '.mov',
            '.mp2', '.mp4', '.mpeg', '.mpeg1video', '.mpeg2video', '.mpeg4', '.mpg',
            '.mts', '.mxf', '.nsv', '.nuv', '.ogg', '.ogv', '.ps', '.rec', '.rm',
            '.rmvb', '.roq', '.svi', '.ts', '.vob', '.webm', '.wmv', '.y4m', '.yuv'
        ]

    def _apply_filters(self, stream, normalize=False, equalizer=None):
        if normalize:
            stream = ffmpeg.filter(stream, 'dynaudnorm')
        if equalizer:
            stream = ffmpeg.filter(stream, 'equalizer', equalizer)
        return stream

    def _analyze_volume(self, input_file):
        try:
            stats = ffmpeg.probe(input_file)
            audio_stats = next((s for s in stats['streams'] if s['codec_type'] == 'audio'), None)
            if audio_stats:
                volume_mean = float(audio_stats['tags']['volume_mean'])
                volume_max = float(audio_stats['tags']['volume_max'])
                return volume_mean, volume_max
            else:
                print("No audio stream found in the input file.")
        except ffmpeg.Error as e:
            print(f"Error occurred during volume analysis: {e.stderr}")
        return None, None

    def _adjust_volume(self, stream, volume_mean, volume_max, target_volume):
        if volume_mean is not None and volume_max is not None:
            volume_adjustment = target_volume - volume_max
            stream = ffmpeg.filter(stream, 'volume', volume=f'{volume_adjustment}dB')
        return stream

    def _convert(self, input_file, output_path, normalize=False, equalizer=None, vbr=False, metadata=None):
        stream = ffmpeg.input(input_file)

        if normalize:
            stream = self._apply_filters(stream, normalize=True)
        else:
            if self.adjust_volume:
                volume_mean, volume_max = self._analyze_volume(input_file)
                if volume_mean is not None and volume_max is not None:
                    stream = self._adjust_volume(stream, volume_mean, volume_max, self.target_volume)

        stream = self._apply_filters(stream, equalizer=equalizer)

        kwargs = {
            'acodec': 'aac',
            'ar': self.sample_rate,
            'ac': self.channels,
        }
        if vbr:
            kwargs['vbr'] = 5
        else:
            kwargs['b:a'] = self.bitrate

        output_stream = ffmpeg.output(stream, output_path, **kwargs)

        try:
            # '-y' オプションを追加して、出力ファイルの自動上書きを許可
            ffmpeg.run(output_stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            print("Conversion completed successfully.")
        except ffmpeg.Error as e:
            stdout = e.stdout.decode('utf-8') if e.stdout else "No stdout"
            stderr = e.stderr.decode('utf-8') if e.stderr else "No stderr"
            print(f"Error occurred during conversion: {stderr}")
            print(f"FFmpeg stdout: {stdout}")

    def convert(self, input_file, output_dir, normalize=True, equalizer=None, vbr=False, metadata=None):
        _, extension = os.path.splitext(input_file)
        if extension.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {extension}")

        output_file = os.path.splitext(os.path.basename(input_file))[0] + ".m4a"
        output_path = os.path.join(output_dir, output_file)
        os.makedirs(output_dir, exist_ok=True)

        self._convert(input_file, output_path, normalize, equalizer, vbr, metadata)
        return output_path

    def __call__(self, input_file, output_dir, normalize=True, equalizer=None, vbr=False, metadata=None):
        return self.convert(input_file, output_dir, normalize, equalizer, vbr, metadata)


#文字おこし
# audio_file_path = os.path.join(folder_path, FILE_NAME)
# json_file_path = os.path.join(folder_path, 'deepgram.json')
# transcript_file_path = os.path.join(folder_path, 'transcript.txt')



st.set_page_config(
    page_title="デモ",
    layout="wide", # wideにすると横長なレイアウトに
    initial_sidebar_state="expanded"
)

st.title("新卒")
st.sidebar.markdown("# STEP 1 ")
interviewee_name = st.sidebar.text_input("担当者の名前を入力してください", placeholder="例:山田太郎")
job_seeker_name = st.sidebar.text_input("求職者の名前を入力してください", placeholder="例:山田花子")
st.sidebar.markdown("# STEP 2 ")
uploaded_file = st.sidebar.file_uploader(
    "動画or音声ファイルをアップロードしてください", type=["mp4", "m4a", "wav"]
)

col1, col2 = st.columns(2)

with col1:
    st.header("Gemini1.5")

with col2:
    st.header("G1.5F")


# def genearate(audio_file, model_name):
#     global price_text
#     genai.configure(api_key=st.secrets.GEMINI_API_KEY.key)
#     audio_file = genai.upload_file(path=audio_file)

#     while audio_file.state.name == "PROCESSING":
#         st.write("wait")
#         time.sleep(10)
#         audio_file = genai.get_file(audio_file.name)

#     if audio_file.state.name == "FAILED":
#         raise ValueError(audio_file.state.name)     

#     model = genai.GenerativeModel(
#         model_name=model_name
#     )

#     prompt = """
# 面談音声の中から、私が指定する項目の内容で、議事録を作成してもらいます。
# ■導入から始めてください。
# ###指定する項目
# ■導入
# ・商談背景(なぜ今回の商談が行われたか)


# ■質問事項
# ・採用背景
# ・採用計画
# ・採用計画に対する進捗状況
# ・採用に伴う課題

# ■採用手法
# ・今までの採用手法
# ・現在利用している他社サービス
# ・他社サービスを利用しての所感

# ■条件
# ・給与
# ・年齢
# ・転職回数
# ・性別
# ・学歴
# ・資格
# ・その他

# ■求める人物像
# ・性格
# ・持ち合わせる経験
# ・活躍する人の傾向

# ■事業内容・業務内容
# ・事業内容
# ・具体的な業務
# ・１日の流れ

# ■会社風土
# ・設立背景
# ・どのような方が活躍するか
# ・会社の雰囲気・文化
# ・他社との違い


# ■選考フロー
# ・選考回数
# ・選考の流れ


# ■契約内容
# ・手数料
# ・返金規定

# ■次回アクション


# ■その他ユニークポイント
# ・他社との違いやアピールポイント等

# ■AI分析

# ・うまく関係値を気づけたか(アイスブレイク)
# ・自社の紹介、他社との違いを説明できたか(自社・自己紹介)
# ・相手の課題をヒアリングできたか(ヒアリング)
# ・サービスをしっかり提案できたいか(サービス提案)
# ・クロージングを行えたか(クロージング)
# ・次回の具体的なアクションを決めれたか(アクション設定)

# ■良かった点

# ■改善点

# ■まとめ

# """
#     input_token = model.count_tokens([prompt, audio_file]).total_tokens
#     response = model.generate_content(
#         [prompt, audio_file], stream=True
#     )

#     genai.delete_file(audio_file.name)



#     output_token = response.usage_metadata.candidates_token_count

#     if model_name == "models/gemini-1.5-pro-latest":
#         input_price = 0.0000035 * input_token
#         output_price = 0.0000105 * output_token
#         price = input_price + output_price
#         price_text = f"\n ## price(USD): ${format(price, '.6f')}"
 
#     if model_name == "models/gemini-1.5-flash-latest":
#         input_price = 0.00000035 * input_token
#         output_price = 0.00000105 * output_token
#         price = input_price + output_price
#         price_text = f"\n ## price(USD): ${format(price, '.6f')}"   

#     return response
def genearate(audio_file, model_name, prompt):
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

    response = model.generate_content(
        [audio, prompt], stream=True
    )


    print(response)
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

    # audio_time = get_audio_duration(m4a_path)
    # st.write(audio_time)

    # queue1 = Queue()
    # queue2 = Queue()
    # queue3 = Queue()
    # future1 = Thread(target=genearate, args=(m4a_path, "models/gemini-1.5-pro-latest", queue1))
    # future2 = Thread(target=genearate, args=(m4a_path, "models/gemini-1.5-flash-latest", queue2))
    # future3 = Thread(target=generate_gpt4o, args=(m4a_path, "gpt-4o-2024-05-13", queue3))



    # future1.start()
    # future2.start()
    # future3.start()


    minutes_container1 = col1.empty()
    minutes_container2 = col2.empty()
    # minutes_container3 = col3.empty()

    prompt1 = f"""
あなたは、{interviewee_name}と{job_seeker_name}の就職活動に関する面談音声から正確で詳細な議事録を作成する専門家です。以下の指示に従って、高品質な議事録を作成してください。

## 一般的な指示
1. {interviewee_name}と{job_seeker_name}の話している内容は正確に分離して、{job_seeker_name}に関連する情報で議事録を作成してください。
2. 提供された音声データを注意深く聞き、内容を正確に理解してください。
3. 議事録は指定された項目に厳密に従って構成してください。
4. 各項目の内容は具体的で詳細なものにしてください。単なる要約ではなく、重要な情報をすべて含めてください。
5. 専門用語や固有名詞は正確に記録してください。不明な場合は、音声をそのまま書き起こし、[不明]とマークしてください。
6. 話者の感情や態度、声のトーンなど、非言語的な情報も適切に記録してください。
7. 議論の流れや重要なポイントが明確になるように、論理的に情報を整理してください。

## 項目別の指示

1. 趣味・趣向
   - {job_seeker_name}の趣味や興味、好みについて詳細に記録してください。金銭的な情報も含めてください。

2. 過去の経験
   - {job_seeker_name}の学生時代の経験や成果、挫折などを時系列で記録してください。

3. 現在での経験
   - {job_seeker_name}の現在の学業状況やアルバイト経験などを具体的に記載してください。

4. 価値基準
   - {job_seeker_name}の価値観や大切にしていることを明確に記録してください。

5. キャリアビジョン・目標
   - {job_seeker_name}の将来のキャリアに関する展望や具体的な目標を詳細に記載してください。

6. 希望勤務地
   - {job_seeker_name}の希望する勤務地や地域に関する情報を記録してください。言及がない場合はその旨を記載します。

7. 企業にもとめること
   - {job_seeker_name}の就職先の企業に求める条件や希望する職種を具体的に記載してください。

8. 選考状況
   - {job_seeker_name}の現在の就職活動の進捗状況や受けている選考について記録してください。

9. 他社エージェント利用状況
   - {job_seeker_name}の他の就職エージェントの利用状況について記載してください。

10. 学生時代での経験
    - {job_seeker_name}の大学生活での特筆すべき経験やプロジェクトについて詳細に記録してください。

11. マッチしそうな企業文化
    - {job_seeker_name}の面談者の性格や価値観に合いそうな企業文化について記載してください。

12. マッチする具体的な企業
    - {job_seeker_name}のエージェントが提案した具体的な企業や職種について記録してください。

13. 現在の心情
    - {job_seeker_name}の就職活動に対する現在の気持ちや心境を詳細に記載してください。

14. 次回アクション
    - {job_seeker_name}の面談後の具体的な行動計画や次のステップについて記録してください。

15. 就活に対する考え方
    - {job_seeker_name}の就職活動全般に対する面談者の考え方や姿勢を記載してください。

16. 企業への推薦文
    - この項目は、就職エージェントが書くことを想定して、あなたが代わりに作成してください。
    - {job_seeker_name}の経験や特性を基に、魅力的で説得力のある推薦文を作成してください。
    - 推薦文は詳細で長めの内容にし、以下の要素を含めてください：
        a) {job_seeker_name}の学歴と主な学業成績
        b) 関連するスキルや能力（特に職務に関連するもの）
        c) 性格特性や仕事への姿勢
        d) 過去の成果や経験（インターンシップ、プロジェクト、アルバイトなど）
        e) {job_seeker_name}が企業にもたらす可能性のある価値
        f) 推薦者（エージェント）の総合的な評価
    - 面談者の名前は記載する必要はないです。

17. エントリーシート自己PR文
    - この項目は、{job_seeker_name}自身が書くことを想定して、あなたが代わりに作成してください。
    - 詳細で長めの自己PR文を作成し、以下の要素を強調してください：
        a) 学歴と主な学業成績
        b) 関連する経験（インターンシップ、プロジェクト、アルバイト）とそこから得た学び
        c) 職務に関連するスキルと能力
        d) 性格特性と仕事への姿勢
        e) キャリアビジョンと志望動機
        f) 自身が企業にもたらす価値
    - 自己PR文は、具体的なエピソードや数値を用いて説得力のある内容にしてください。
    - 文体は丁寧で前向きなトーンを維持し、読み手に強い印象を与えるようにしてください。
    - {job_seeker_name}の名前は記載する必要はないです。

## 出力フォーマット

議事録は以下のフォーマットで出力してください：

```
[面談議事録]

1. 趣味・趣向
   [内容]

2. 過去の経験
   [内容]

3. 現在での経験
   [内容]

4. 価値基準
   [内容]

5. キャリアビジョン・目標
   [内容]

6. 希望勤務地
   [内容]

7. 企業にもとめること
   [内容]

8. 選考状況
   [内容]

9. 他社エージェント利用状況
   [内容]

10. 学生時代での経験
    [内容]

11. マッチしそうな企業文化
    [内容]

12. マッチする具体的な企業
    [内容]

13. 現在の心情
    [内容]

14. 次回アクション
    [内容]

15. 就活に対する考え方
    [内容]

16. 企業への推薦文
    [内容]

17. エントリーシート自己PR文
    [内容]
```

この指示に従って、正確で詳細、かつ有用な面談議事録を作成してください。不明な点がある場合は、必ず確認を求めてください。

    """

    prompt2 = """
あなたは、就職活動に関する面談音声から正確で詳細な分析を行い評価する専門家です。今回は、キャリアアドバイザーの面談を分析し、評価を行っていただきます。以下の指示に従って、高品質な分析評価レポートを作成してください。

## 一般的な指示

1. 提供された音声データを注意深く聞き、内容を正確に理解してください。
2. 議事録は指定された項目に厳密に従って構成してください。
3. 各項目の内容は具体的で詳細なものにしてください。単なる要約ではなく、重要な情報をすべて含めてください。
4. 専門用語や固有名詞は正確に記録してください。不明な場合は、音声をそのまま書き起こし、[不明]とマークしてください。
5. 話者の感情や態度、声のトーンなど、非言語的な情報も適切に記録してください。
6. 議論の流れや重要なポイントが明確になるように、論理的に情報を整理してください。

## 項目別の指示
キャリアアドバイザーを評価する。
■AI分析

・うまく関係値を気づけたか(アイスブレイク)
・自社の紹介、他社との違いを説明できたか(自社・自己紹介)
・相手の課題をヒアリングできたか(ヒアリング)
・サービスをしっかり提案できたいか(サービス提案)
・クロージングを行えたか(クロージング)
・次回の具体的なアクションを決めれたか(アクション設定)

■良かった点

■改善点

■まとめ
"""    




    with st.spinner('Gemini 1.5 (pro) 実行中...'):
        minutes1 = ""
        response1 = genearate(m4a_path, "gemini-pro-experimental", prompt1)
        for chunk1 in response1:
            minutes1 += chunk1.text
            minutes_container1.write(minutes1)
            time.sleep(0.05)
        # response2 = genearate(m4a_path, "models/gemini-1.5-pro", prompt2)
        # for chunk2 in response2:
        #     minutes1 += chunk2.text
        #     minutes_container1.write(minutes1)
        #     time.sleep(0.05)
        minutes1 += price_text
        minutes_container1.write(minutes1)

    with st.spinner('Gemini 1.5 (flash) 実行中...'):
        minutes2 = ""
        response3 = genearate(m4a_path, "gemini-flash-experimental", prompt1)
        for chunk3 in response3:
            minutes2 += chunk3.text
            minutes_container2.write(minutes2)
            time.sleep(0.05)

        # response4 = genearate(m4a_path, "models/gemini-1.5-pro", prompt2)
        # for chunk4 in response4:
        #     minutes2 += chunk4.text
        #     minutes_container2.write(minutes2)
        #     time.sleep(0.05)    
        minutes2 += price_text
        minutes_container2.write(minutes2)

    # with st.spinner('GPT-4o 実行中...'):
    #     minutes3 = generate_gpt4o(m4a_path, "gpt-4o-2024-05-13")
    #     minutes_container3.write(minutes3)

    



    # while future1.is_alive() or future2.is_alive():
    #     if not queue1.empty():
    #         minutes_container1.write(queue1.get())
    #     if not queue2.empty():
    #         minutes_container2.write(queue2.get()) 
    #     if not queue3.empty():
    #         minutes_container3.write(queue3.get()) 
    # future1.join()
    # future2.join()
    # future3.join()



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

    # tran = transcribe(m4a_path)
    # res = gpt4o_generate(tran, "gpt-4o-2024-05-13")
    # minutes_container3.write(res)


    

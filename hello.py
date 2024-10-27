import asyncio
import websockets
import pyaudio
import numpy as np
import base64
import json
import queue
import threading
import os
import time
import resampy

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get('OPENAI_API_KEY')

# WebSocket URLとヘッダー情報
# OpenAI
WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
HEADERS = {
    "Authorization": "Bearer " + API_KEY,
    "OpenAI-Beta": "realtime=v1"
}

# キューを初期化
audio_send_queue = queue.Queue()
audio_receive_queue = queue.Queue()

# PyAudioの設定
INPUT_CHUNK = 1024
OUTPUT_CHUNK = 1024
FORMAT = pyaudio.paInt16

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 2
INPUT_RATE = 44100  # マイクのサンプリングレートに合わせてください
OUTPUT_RATE = 48000  # 出力デバイスのサンプリングレートに合わせてください

# デバイスIDの設定（必要に応じて変更）
INPUT_DEVICE_INDEX = None
OUTPUT_DEVICE_INDEX = None

# PCM16形式に変換する関数
def base64_to_pcm16(base64_audio):
    audio_data = base64.b64decode(base64_audio)
    return audio_data

def resample_audio(audio_data, original_rate, target_rate):
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    audio_resampled = resampy.resample(audio_array, sr_orig=original_rate, sr_new=target_rate)
    audio_resampled_int16 = audio_resampled.astype(np.int16)
    return audio_resampled_int16.tobytes()

# 音声を送信する非同期関数
async def send_audio_from_queue(websocket):
    while True:
        audio_data = await asyncio.get_event_loop().run_in_executor(None, audio_send_queue.get)
        if audio_data is None:
            continue

        # リサンプリング
        if INPUT_RATE != 24000:
            audio_data = resample_audio(audio_data, original_rate=INPUT_RATE, target_rate=24000)

        # 音量を調整
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio_array *= 0.5  # 必要に応じて調整
        audio_array = np.clip(audio_array, -32768, 32767)
        audio_data = audio_array.astype(np.int16).tobytes()

        # PCM16データをBase64にエンコード
        base64_audio = base64.b64encode(audio_data).decode("utf-8")

        audio_event = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }

        # WebSocketで音声データを送信
        await websocket.send(json.dumps(audio_event))

        # キューの処理間隔を少し空ける
        await asyncio.sleep(0)

# マイクからの音声を取得しキューに入れる関数
def read_audio_to_queue(stream, CHUNK):
    while True:
        try:
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
            audio_send_queue.put(audio_data)
        except Exception as e:
            print(f"音声読み取りエラー: {e}")
            break

# サーバーから音声を受信してキューに格納する非同期関数
async def receive_audio_to_queue(websocket):
    print("assistant: ", end = "", flush = True)
    while True:
        response = await websocket.recv()
        if response:
            response_data = json.loads(response)

            # サーバーからの応答をリアルタイムに表示
            if "type" in response_data and response_data["type"] == "response.audio_transcript.delta":
                print(response_data["delta"], end = "", flush = True)
            # サーバからの応答が完了したことを取得
            elif "type" in response_data and response_data["type"] == "response.audio_transcript.done":
                print("\nassistant: ", end = "", flush = True)

            #発話開始の検知
            if "type" in response_data and response_data["type"] == "input_audio_buffer.speech_started":
                # 既存の音声データをクリア
                while not audio_receive_queue.empty():
                    audio_receive_queue.get() 

            # サーバーからの音声データをキューに格納
            if "type" in response_data and response_data["type"] == "response.audio.delta":
                base64_audio_response = response_data["delta"]
                if base64_audio_response:
                    pcm16_audio = base64_to_pcm16(base64_audio_response)
                    audio_receive_queue.put(pcm16_audio)
                    
        await asyncio.sleep(0)

# サーバーからの音声を再生する関数
def play_audio_from_queue(output_stream):
    while True:
        pcm16_audio = audio_receive_queue.get()
        if pcm16_audio:
            # リサンプリング
            if OUTPUT_RATE != 24000:
                pcm16_audio = resample_audio(pcm16_audio, original_rate=24000, target_rate=OUTPUT_RATE)

            # numpy 配列に変換
            audio_array = np.frombuffer(pcm16_audio, dtype=np.int16).astype(np.float32)

            # 音量を調整
            audio_array *= 0.5  # 必要に応じて調整
            audio_array = np.clip(audio_array, -32768, 32767)

            # int16 に再変換
            audio_array = audio_array.astype(np.int16)

            # チャンネル数を変換（モノラルからステレオ）
            if OUTPUT_CHANNELS == 2:
                stereo_array = np.column_stack((audio_array, audio_array)).flatten()
                pcm16_audio = stereo_array.tobytes()
            else:
                pcm16_audio = audio_array.tobytes()

            # 音声を再生
            output_stream.write(pcm16_audio)

# マイクからの音声を取得し、WebSocketで送信しながらサーバーからの音声応答を再生する非同期関数
async def stream_audio_and_receive_response():
    # WebSocketに接続
    async with websockets.connect(WS_URL, extra_headers=HEADERS) as websocket:
        print("WebSocketに接続しました。")

        update_request = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": "日本語かつ関西弁で回答してください。",
                "voice": "alloy",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                }
            }
        }
        await websocket.send(json.dumps(update_request))

        # PyAudioインスタンス
        p = pyaudio.PyAudio()

        # マイクストリームの初期化
        stream = p.open(format=FORMAT, channels=INPUT_CHANNELS, rate=INPUT_RATE, input=True,
                        frames_per_buffer=INPUT_CHUNK, input_device_index=INPUT_DEVICE_INDEX)

        # サーバーからの応答音声を再生するためのストリームを初期化
        output_stream = p.open(format=FORMAT, channels=OUTPUT_CHANNELS, rate=OUTPUT_RATE, output=True,
                               frames_per_buffer=OUTPUT_CHUNK, output_device_index=OUTPUT_DEVICE_INDEX)

        # マイクの音声読み取りをスレッドで開始
        threading.Thread(target=read_audio_to_queue, args=(stream, INPUT_CHUNK), daemon=True).start()

        # サーバーからの音声再生をスレッドで開始
        threading.Thread(target=play_audio_from_queue, args=(output_stream,), daemon=True).start()

        try:
            # 音声送信タスクと音声受信タスクを非同期で並行実行
            send_task = asyncio.create_task(send_audio_from_queue(websocket))
            receive_task = asyncio.create_task(receive_audio_to_queue(websocket))

            # タスクが終了するまで待機
            await asyncio.gather(send_task, receive_task)

        except KeyboardInterrupt:
            print("終了します...")
        finally:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
            output_stream.stop_stream()
            output_stream.close()
            p.terminate()

if __name__ == "__main__":
    asyncio.run(stream_audio_and_receive_response())
import pyaudio

p = pyaudio.PyAudio()

print("使用可能なオーディオデバイス一覧:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"ID: {info['index']}, 名前: {info['name']}")
    print(f"  入力チャンネル: {info['maxInputChannels']}, 出力チャンネル: {info['maxOutputChannels']}")
    print(f"  デフォルトのサンプルレート: {info['defaultSampleRate']}")
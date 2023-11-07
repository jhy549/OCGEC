from torchaudio import datasets
import os


datasets.SPEECHCOMMANDS(
    root="../raw_data",                         # 你保存数据的路径
    url = 'speech_commands_v0.02',         # 下载数据版本URL
    folder_in_archive = 'speech_command',
    download = True                        # 这个记得选True
)
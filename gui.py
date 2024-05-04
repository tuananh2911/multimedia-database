import tkinter as tk
from tkinter import filedialog, messagebox
import csv
import json
import os
import pygame
import numpy as np

from detect import AudioAnalyzer


class AudioAnalyzerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Audio Analyzer")

        self.file_label = tk.Label(master, text="Chọn file âm thanh:")
        self.file_label.pack()

        self.choose_file_button = tk.Button(master, text="Chọn File", command=self.choose_file)
        self.choose_file_button.pack()

        self.table = tk.Label(master, text="Thông tin các file gần nhất sẽ xuất hiện ở đây.")
        self.table.pack()

        self.play_button = tk.Button(master, text="Phát", command=self.play_selected_audio)
        self.play_button.pack()

        self.selected_file = None
        self.analyzer = None

    def choose_file(self):
        self.selected_file = filedialog.askopenfilename(initialdir=os.getcwd(), title="Chọn File",
                                                        filetypes=(("Audio files", "*.wav;*.mp3"), ("All files", "*.*")))
        if self.selected_file:
            self.analyzer = AudioAnalyzer("audio_features.json")
            self.display_similar_files()

    def display_similar_files(self):
        top_similar_files = self.analyzer.find_most_similar_segments(self.selected_file)
        table_content = ""
        for idx, data in enumerate(top_similar_files):
            table_content += f"{idx+1}. {data['file_name']} - Độ tương đồng: {data['average_distance']:.2f}\n"
        self.table.config(text=table_content)

    def play_selected_audio(self):
        if self.selected_file:
            pygame.init()
            pygame.mixer.init()
            pygame.mixer.music.load(self.selected_file)
            pygame.mixer.music.play()

def main():
    root = tk.Tk()
    app = AudioAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

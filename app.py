import tkinter as tk
from config import BG

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Women's Health Analyzer")
        self.root.configure(bg=BG)
        self.root.geometry("900x600")

        label = tk.Label(self.root, text="Welcome to Women's Health Analyzer", bg=BG, fg="white")
        label.pack(pady=20)

    def run(self):
        self.root.mainloop()

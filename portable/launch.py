"""Entry for 매의눈.exe, a renamed copy of the signed pythonw.exe; Python runs this at startup from Lib/site-packages/maeui.pth."""
import os
import runpy
import sys
import traceback
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0,str(APP))
(APP/"outputs").mkdir(exist_ok=True)
sys.stdout = sys.stderr = open(APP/"outputs/error_log.txt","a",encoding="utf-8")  # pythonw has no console
try:
    runpy.run_path(str(APP/"app.py"),run_name="__main__")
except Exception:
    traceback.print_exc()
    from tkinter import messagebox
    messagebox.showerror("매의 눈",traceback.format_exc())
sys.stderr.flush()
os._exit(0)  # otherwise Python carries on into its interactive prompt

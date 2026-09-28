import sys
import os

def choose_folder():
    # Metodo 1: Tkinter en proceso dedicado (corre en el hilo principal)
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        root.focus_force()
        folder = filedialog.askdirectory(title="Selecciona la carpeta con imágenes de prueba")
        root.destroy()
        if folder:
            return os.path.normpath(os.path.abspath(folder))
    except Exception:
        pass

    # Metodo 2: PowerShell Windows Forms nativo
    try:
        import subprocess
        ps_code = """
        [System.Reflection.Assembly]::LoadWithPartialName("System.windows.forms") | Out-Null
        $f = New-Object System.Windows.Forms.FolderBrowserDialog
        $f.Description = "Selecciona la carpeta con imágenes de prueba"
        $f.ShowNewFolderButton = $false
        if ($f.ShowDialog((New-Object System.Windows.Forms.NativeWindow)) -eq "OK") {
            [Console]::Write($f.SelectedPath)
        }
        """
        res = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_code], capture_output=True, text=True, timeout=60)
        out = res.stdout.strip()
        if out and os.path.isdir(out):
            return os.path.normpath(os.path.abspath(out))
    except Exception:
        pass

    return ""

if __name__ == '__main__':
    selected = choose_folder()
    if selected:
        print(selected)

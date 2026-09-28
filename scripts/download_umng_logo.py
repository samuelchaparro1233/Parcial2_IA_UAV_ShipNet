import urllib.request
import os

os.makedirs('assets', exist_ok=True)
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

urls = [
    ('assets/logo_umng.png', 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Escudo_oficial_Universidad_Militar_Nueva_Granada.svg/500px-Escudo_oficial_Universidad_Militar_Nueva_Granada.svg.png'),
    ('assets/logo_umng.svg', 'https://upload.wikimedia.org/wikipedia/commons/d/d2/Escudo_oficial_Universidad_Militar_Nueva_Granada.svg')
]

for dest, url in urls:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            with open(dest, 'wb') as f:
                f.write(data)
        print(f"Descargado con éxito: {dest} ({len(data)} bytes)")
    except Exception as e:
        print(f"Fallo en {dest}: {e}")

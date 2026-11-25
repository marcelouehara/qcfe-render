"""Título do post e gancho na tela com Gemini Flash (nível grátis da API), a partir da fala e de 3 quadros.

Chave em GEMINI_API_KEY. Sem chave, ou se todos os modelos falharem, devolve None e o corte segue com o
título que o espectador deu ao clipe na Twitch.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
# em ordem; o nível grátis às vezes devolve 503 (alta demanda) ou 429 (limite): tenta o próximo
MODELOS = ["gemini-3.5-flash", "gemini-flash-latest", "gemini-flash-lite-latest"]

PROMPT = """Voce escreve titulos de Shorts/TikTok para um canal brasileiro de cortes de GTA 6.
Streamer: {streamer}.
Titulo que um espectador deu ao clipe na Twitch (pode ser lixo, ignore se nao fizer sentido): {twitch!r}
Fala do streamer no corte (transcricao automatica, pode ter erros; pode estar vazia):
\"\"\"{fala}\"\"\"
As imagens sao quadros do corte (em cima a camera do streamer, embaixo o jogo).

Escreva em português do Brasil, com acentuação e ortografia corretas (ex.: "vê", "relação", "não"):
- titulo: ate 60 caracteres, chamativo, com o nome do streamer, dizendo o que acontece. So use o que esta na
  fala ou nas imagens. Sem inventar, sem palavrao, sem hashtag, sem emoji.
- gancho: 2 a 5 palavras em MAIUSCULAS para aparecer no video nos primeiros segundos (curiosidade, sem spoiler).
Responda so JSON: {{"titulo": "...", "gancho": "..."}}"""


def quadros(video: str, duracao: float) -> list[bytes]:
    out = []
    for f in (0.25, 0.5, 0.75):
        r = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{duracao * f:.2f}", "-i", video, "-frames:v", "1",
                            "-vf", "scale=512:-2", "-f", "image2", "-c:v", "mjpeg", "-"], capture_output=True)
        if r.returncode == 0 and r.stdout:
            out.append(r.stdout)
    return out


def gerar(streamer: str, titulo_twitch: str, fala: str, imagens: list[bytes]) -> dict | None:
    chave = os.environ.get("GEMINI_API_KEY")
    if not chave:
        return None
    partes = [{"text": PROMPT.format(streamer=streamer, twitch=titulo_twitch or "", fala=fala[:3000])}]
    partes += [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(i).decode()}} for i in imagens]
    corpo = json.dumps({"contents": [{"parts": partes}],
                        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.7}}).encode()
    for modelo in MODELOS:
        for tentativa in range(2):
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent", data=corpo,
                headers={"x-goog-api-key": chave, "Content-Type": "application/json"})
            try:
                r = json.load(urllib.request.urlopen(req, timeout=90))
                d = json.loads(r["candidates"][0]["content"]["parts"][0]["text"])
                titulo = re.sub(r"\s+", " ", str(d.get("titulo", ""))).strip().strip('"')
                gancho = re.sub(r"\s+", " ", str(d.get("gancho", ""))).strip().strip('"').upper()
                if 8 <= len(titulo) <= 80 and 3 <= len(gancho) <= 45:
                    return {"titulo": titulo[:70], "gancho": gancho, "modelo": modelo}
                break  # resposta fora do formato: tenta o próximo modelo
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 503) and tentativa == 0:
                    time.sleep(6)
                    continue
                break
            except Exception:  # noqa: BLE001 - JSON quebrado, timeout etc.: tenta o próximo modelo
                break
    return None

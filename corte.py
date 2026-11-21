"""Escolhe e renderiza 1 corte a partir dos clipes candidatos que o n8n manda.

Lê o job da variável JOB (JSON enviado pelo n8n via repository_dispatch):
{"id": "20261121-1400-123", "teste": false,
 "candidatos": [{"clip_id": "...", "url": "https://www.twitch.tv/x/clip/...", "title": "...", "views": 100,
                 "streamer": "hayashii", "cam": "1436,124,450,250", "layout": "cam_top", "gx": 960}, ...]}

Para cada candidato, na ordem: baixa o clipe, descarta se for momento "só câmera" (rosto grande na tela,
sem gameplay) e renderiza o primeiro que passar. Escreve out/<id>.json com o resultado e
out/historico_add.json com os ids de clipe que não devem ser tentados de novo.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).parent
MODEL = str(HERE / "models" / "face_detection_yunet_2023mar.onnx")
FACE_LIMIT = 0.12  # rosto mais largo que 12% da tela = câmera em tela cheia
ALVO, MINIMO, MAXIMO = 75.0, 65.0, 90.0  # duração do corte em segundos (TikTok só monetiza acima de 1 min)


def janela(offset: float, dur: float) -> tuple[float, float]:
    """Trecho da VOD em volta do clipe: ~70% do contexto extra antes (o setup) e ~30% depois (a reação)."""
    extra = max(0.0, ALVO - dur)
    ini = max(0.0, offset - extra * 0.7)
    fim = offset + dur + extra * 0.3
    if fim - ini < MINIMO:
        fim = ini + MINIMO
    return ini, min(fim, ini + MAXIMO)


def baixar(c: dict, destino: str) -> str | None:
    """Baixa o trecho estendido da VOD. Sem VOD pública não dá para passar de 1 min: devolve o motivo."""
    if not c.get("video_id") or c.get("vod_offset") is None:
        return "sem_vod"
    ini, fim = janela(float(c["vod_offset"]), float(c.get("duration") or 30))
    r = subprocess.run(["yt-dlp", "-q", "--no-warnings", "-f", "best[height<=1080]", "--force-overwrites",
                        "--download-sections", f"*{ini:.0f}-{fim:.0f}", "-o", destino,
                        f"https://www.twitch.tv/videos/{c['video_id']}"], capture_output=True, text=True)
    if r.returncode != 0 or not Path(destino).exists():
        return "vod_inscritos" if "subscriber" in r.stderr else "download"
    return None


def so_camera(video: str) -> tuple[bool, list[float]]:
    """True quando a maioria dos quadros amostrados tem um rosto grande (live em modo só câmera)."""
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    det = None
    larguras = []
    for frac in (0.2, 0.4, 0.6, 0.8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = 640 / w
        small = cv2.resize(frame, (640, int(h * scale)))
        if det is None:
            det = cv2.FaceDetectorYN.create(MODEL, "", (small.shape[1], small.shape[0]), 0.7)
        det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = det.detect(small)
        maior = max((f[2] for f in faces), default=0) / small.shape[1] if faces is not None else 0.0
        larguras.append(round(float(maior), 3))
    cap.release()
    grandes = sum(x > FACE_LIMIT for x in larguras)
    return grandes * 2 > len(larguras), larguras


def titulo(t: str, streamer: str) -> str:
    """Título do clipe na Twitch (escrito por quem clipou); se for vazio ou só símbolo, usa um padrão."""
    t = re.sub(r"\s+", " ", t or "").strip()
    if len(re.sub(r"[^A-Za-zÀ-ÿ]", "", t)) < 4:
        return f"{streamer} no GTA 6"
    return t[:90]


def main():
    job = json.loads(os.environ["JOB"])
    jid = job["id"]
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    pulados, escolhido = [], None
    for c in job["candidatos"]:
        src = f"/tmp/clip_{c['clip_id'][:40]}.mp4"
        erro = baixar(c, src)
        if erro:
            print(c["clip_id"], c["streamer"], "->", erro, flush=True)
            pulados.append({"clip_id": c["clip_id"], "motivo": erro})
            continue
        cam, larg = so_camera(src)
        print(c["clip_id"], c["streamer"], "rostos:", larg, "-> so camera" if cam else "-> ok", flush=True)
        if cam:
            pulados.append({"clip_id": c["clip_id"], "motivo": "so_camera"})
            continue
        t = titulo(c.get("title", ""), c["streamer"])
        subprocess.run([sys.executable, str(HERE / "render.py"), src, str(out / f"{jid}.mp4"),
                        "--cam", c["cam"], "--layout", c.get("layout", "cam_top"), "--gx", str(c.get("gx", 960)),
                        "--title", t, "--credit", c["streamer"]], check=True)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(out / f"{jid}.mp4")], capture_output=True, text=True).stdout.strip() or 0)
        escolhido = {**c, "title": t, "duration": dur}
        break
    res = {"id": jid, "status": "ok" if escolhido else "sem_corte", "teste": job.get("teste", False),
           "clip": escolhido, "pulados": pulados}
    (out / f"{jid}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    # descartes definitivos não voltam como candidato (sem_vod pode ser VOD ainda processando: tenta de novo)
    novos = [p["clip_id"] for p in pulados if p["motivo"] in ("so_camera", "vod_inscritos")]
    if escolhido and not job.get("teste"):
        novos.append(escolhido["clip_id"])
    (out / "historico_add.json").write_text(json.dumps(novos))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()

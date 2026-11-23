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
    return grandes * 2 >= len(larguras), larguras  # metade ou mais dos quadros só com a câmera: descarta


def achar_cam(video: str, dica: str | None) -> str | None:
    """Acha a facecam pelo rosto que fica parado no mesmo lugar (os do jogo se mexem). Devolve 'x,y,w,h' num
    quadro 1920x1080, ou None se não houver cam. Se a posição do streamers.json bate com o rosto, usa ela."""
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fw = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
    k = 1920 / 640
    det, quadros = None, []
    for i in range(1, 11):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * i / 11))
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        if det is None:
            det = cv2.FaceDetectorYN.create(MODEL, "", (small.shape[1], small.shape[0]), 0.7)
        det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = det.detect(small)
        quadros.append([(f[0] * k + f[2] * k / 2, f[1] * k + f[3] * k / 2, f[2] * k) for f in (faces if faces is not None else [])
                        if f[2] / 640 < FACE_LIMIT])
    cap.release()
    if not quadros:
        return None
    dentro = lambda cx, cy: False
    if dica:
        dx, dy, dw, dh = (float(v) for v in dica.split(","))
        dentro = lambda cx, cy: dx <= cx <= dx + dw and dy <= cy <= dy + dh
    melhor, apoio = None, []
    for q in quadros:
        for cx, cy, w in q:
            perto = [min(((x, y, ww) for x, y, ww in o), key=lambda f: (f[0] - cx) ** 2 + (f[1] - cy) ** 2, default=None)
                     for o in quadros]
            perto = [f for f in perto if f and abs(f[0] - cx) < 0.6 * w and abs(f[1] - cy) < 0.6 * w]
            if len(perto) > len(apoio):
                melhor, apoio = (cx, cy, w), perto
    if not melhor or len(apoio) < 0.6 * len(quadros):
        # rosto difícil de detectar (cam escura, longe): vale a posição medida se ele apareceu nela ao menos 1 vez
        if any(dentro(cx, cy) for q in quadros for cx, cy, _ in q):
            return dica
        return None
    med = lambda i: sorted(f[i] for f in apoio)[len(apoio) // 2]
    cx, cy, w = med(0), med(1), med(2)
    if dica and dentro(cx, cy) and 2.0 * w <= dw <= 10 * w:
        return dica
    bw = min(4.4 * w, 900)
    bh = bw * 0.58
    x = min(max(cx - bw / 2, 0), 1920 - bw)
    y = min(max(cy - bh * 0.45, 0), 1080 - bh)
    return f"{x:.0f},{y:.0f},{bw:.0f},{bh:.0f}"


def titulo(t: str, streamer: str) -> str:
    """Título do clipe na Twitch (escrito por quem clipou); se for vazio ou só símbolo, usa um padrão."""
    t = re.sub(r"\s+", " ", t or "").strip()
    if len(re.sub(r"[^A-Za-zÀ-ÿ]", "", t)) < 4:
        return f"{streamer} no GTA 6"
    return t[:90]


def duracao(p) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def renderizar(src, destino, c, t):
    cam = achar_cam(src, c.get("cam"))
    layout = c.get("layout", "cam_top") if cam else "full"
    print("    cam:", cam or "nenhuma (layout so gameplay)", flush=True)
    ia_json = Path(destino).with_suffix(".ia.json")
    ia_json.unlink(missing_ok=True)
    subprocess.run([sys.executable, str(HERE / "render.py"), src, str(destino),
                    "--cam", cam or "0,0,1,1", "--layout", layout, "--gx", str(c.get("gx", 960)),
                    "--title", t, "--credit", c["streamer"], "--ia"], check=True)
    return json.loads(ia_json.read_text()) if ia_json.exists() else None


def compilacao(c: dict, destino: Path, pulados: list) -> dict | None:
    """Sem VOD pública o clipe fica abaixo de 1 min: junta de 2 a 4 clipes do mesmo streamer (cada um com o
    título dele no começo) até passar de MINIMO segundos."""
    partes, total, ia_principal = [], 0.0, None
    for i, x in enumerate([c] + c.get("extras", [])):
        x = {**c, **x}  # extras herdam cam/layout/gx do streamer
        if partes and total + float(x.get("duration") or 30) > 120:  # compilação de no máximo ~2 min
            continue
        src = f"/tmp/comp_{i}.mp4"
        r = subprocess.run(["yt-dlp", "-q", "--no-warnings", "-f", "best[height<=1080]", "--force-overwrites",
                            "-o", src, x["url"]], capture_output=True, text=True)
        if r.returncode != 0 or not Path(src).exists():
            continue
        cam, larg = so_camera(src)
        print("  compilacao:", x["clip_id"], "rostos:", larg, "-> so camera" if cam else "-> ok", flush=True)
        if cam:
            pulados.append({"clip_id": x["clip_id"], "motivo": "so_camera"})
            continue
        seg = Path(f"/tmp/seg_{i}.mp4")
        ia = renderizar(src, seg, x, titulo(x.get("title", ""), x["streamer"]))
        if not partes:
            ia_principal = ia  # o título do post vem do primeiro trecho da compilação
        partes.append((x, seg))
        total += duracao(seg)
        if total >= MINIMO + 5 or len(partes) >= 4:
            break
    if total < MINIMO or len(partes) < 2:
        return None
    lista = Path("/tmp/comp.txt")
    lista.write_text("".join(f"file '{seg}'\n" for _, seg in partes))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lista),
                    "-c", "copy", "-movflags", "+faststart", str(destino)], check=True)
    return {**c, "title": f"{c['streamer']} no GTA 6: {len(partes)} momentos", "duration": duracao(destino),
            "ia": ia_principal,
            "views": sum(int(x.get("views") or 0) for x, _ in partes),
            "compilacao": [x["clip_id"] for x, _ in partes]}


def textos(c: dict, titulo_principal: str) -> dict:
    """Título e legenda dos posts (fonte única: o n8n só repassa ao Buffer)."""
    s = c["streamer"]
    nome = s[:1].upper() + s[1:]
    comp = len(c.get("compilacao") or []) > 1
    generico = titulo_principal.endswith("no GTA 6")
    t = f"{nome} no GTA 6: {len(c['compilacao'])} momentos da live" if comp and generico else titulo_principal
    if (c.get("ia") or {}).get("titulo"):  # título do Gemini, quando deu certo
        t = c["ia"]["titulo"]
    tag = re.sub(r"[^a-z0-9]", "", s.lower())
    legenda = (f"{t}\n\n{'Cortes' if comp else 'Corte'} da live do {s} na Twitch (twitch.tv/{s})\n\n"
               f"#gta6 #gtavi #gta #{tag} #cortes")
    titulo_yt = t if "gta" in t.lower() else f"{t} | {nome} GTA 6"
    return {"titulo": t, "legenda": legenda, "titulo_yt": titulo_yt[:100], "descricao_yt": legenda + " #shorts",
            "tags": ["gta 6", "gta vi", "gta6", "cortes", s]}


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
            print(c["clip_id"], c["streamer"], "->", erro, "-> tentando compilacao", flush=True)
            escolhido = compilacao(c, out / f"{jid}.mp4", pulados)
            if escolhido:
                break
            pulados.append({"clip_id": c["clip_id"], "motivo": erro})
            continue
        cam, larg = so_camera(src)
        print(c["clip_id"], c["streamer"], "rostos:", larg, "-> so camera" if cam else "-> ok", flush=True)
        if cam:
            pulados.append({"clip_id": c["clip_id"], "motivo": "so_camera"})
            continue
        t = titulo(c.get("title", ""), c["streamer"])
        ia = renderizar(src, out / f"{jid}.mp4", c, t)
        escolhido = {**c, "title": t, "duration": duracao(out / f"{jid}.mp4"), "ia": ia}
        break
    posts, yt = None, None
    if escolhido:
        escolhido.pop("extras", None)
        cand = next(c for c in job["candidatos"] if c["clip_id"] == escolhido["clip_id"])
        posts = textos(escolhido, titulo(cand.get("title", ""), escolhido["streamer"]))
        # YouTube pela API (cota ~6 uploads/dia); se falhar (ex.: quotaExceeded) o n8n manda pelo Buffer
        if not job.get("teste") and job.get("youtube_api", True) and os.environ.get("YT_REFRESH_TOKEN"):
            import youtube
            yt = youtube.upload(str(out / f"{jid}.mp4"), posts["titulo_yt"], posts["descricao_yt"], posts["tags"],
                                job.get("publicar_em"))
            print("youtube:", yt, flush=True)
    res = {"id": jid, "status": "ok" if escolhido else "sem_corte", "teste": job.get("teste", False),
           "clip": escolhido, "posts": posts, "youtube": yt, "pulados": pulados}
    (out / f"{jid}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    # descartes definitivos não voltam como candidato (sem_vod pode ser VOD ainda processando: tenta de novo)
    novos = [p["clip_id"] for p in pulados if p["motivo"] in ("so_camera", "vod_inscritos")]
    if escolhido and not job.get("teste"):
        novos += escolhido.get("compilacao") or [escolhido["clip_id"]]
    (out / "historico_add.json").write_text(json.dumps(novos))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()

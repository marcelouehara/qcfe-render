"""Monta um corte vertical 1080x1920: cam do streamer + gameplay empilhados, legenda palavra a palavra.

Uso:
  python render.py ORIGEM.mp4 SAIDA.mp4 --cam 1436,124,450,250 --title "Ele encontrou o tigre" --credit hayashii
Opções:
  --cam x,y,w,h      posição da facecam no vídeo original, em pixels de um quadro 1920x1080
  --layout           cam_top (padrão) ou cam_bottom
  --gx               centro horizontal do recorte do gameplay em 1920x1080 (padrão: meio da tela)
  --model            modelo do whisper (padrão small; base é mais rápido, medium mais preciso)
A legenda é desenhada com Pillow em PNGs transparentes e sobreposta pelo ffmpeg (não precisa de libass).
Variáveis: FFMPEG / FFPROBE apontam para os binários.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
REF_W, REF_H = 1920, 1080
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")
FONT = str(Path(__file__).parent / "fonts" / "Anton-Regular.ttf")
WHITE, YELLOW, BLACK = (255, 255, 255, 255), (255, 229, 0, 255), (0, 0, 0, 255)
TITLE_SECONDS = 4.0


def probe(src: str) -> tuple[int, int, float]:
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", src],
                         capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    s = d["streams"][0]
    return int(s["width"]), int(s["height"]), float(d["format"]["duration"])


def transcribe(src: str, model_name: str) -> list[dict]:
    import numpy as np
    from faster_whisper import WhisperModel

    # o ffmpeg decodifica o audio (evita depender do PyAV do faster-whisper)
    pcm = subprocess.run([FFMPEG, "-v", "error", "-i", src, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(audio, language="pt", word_timestamps=True, vad_filter=True)
    words = []
    for seg in segs:
        for w in seg.words or []:
            t = w.word.strip()
            if t:
                words.append({"start": w.start, "end": w.end, "text": t})
    return words


def groups(words: list[dict], size: int = 3, gap: float = 0.6) -> list[list[dict]]:
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= size or w["start"] - cur[-1]["end"] > gap):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def caption_events(words: list[dict]) -> list[tuple[float, float, list[str], int]]:
    """(início, fim, palavras do grupo, índice da palavra destacada)"""
    ev = []
    for g in groups(words):
        for i, w in enumerate(g):
            end = g[i + 1]["start"] if i + 1 < len(g) else g[-1]["end"] + 0.15
            ev.append((w["start"], max(end, w["start"] + 0.05), [x["text"].upper() for x in g], i))
    return ev


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def draw_caption(d: ImageDraw.ImageDraw, words: list[str], hi: int, y: int):
    size = 96
    while size > 50:
        f = font(size)
        space = d.textlength(" ", font=f)
        widths = [d.textlength(w, font=f) for w in words]
        total = sum(widths) + space * (len(words) - 1)
        if total <= W - 80:
            break
        size -= 6
    x = (W - total) / 2
    for i, (w, wd) in enumerate(zip(words, widths)):
        d.text((x, y), w, font=f, fill=YELLOW if i == hi else WHITE, stroke_width=8, stroke_fill=BLACK, anchor="lm")
        x += wd + space


def draw_title(d: ImageDraw.ImageDraw, title: str, y: int):
    f = font(60)
    lines = textwrap.wrap(title.upper(), 24)[:3]
    lh = 74
    top = y - lh * len(lines) / 2
    for i, line in enumerate(lines):
        tw = d.textlength(line, font=f)
        cy = top + lh * i + lh / 2
        d.rectangle([(W - tw) / 2 - 22, cy - lh / 2, (W + tw) / 2 + 22, cy + lh / 2], fill=YELLOW)
        d.text((W / 2, cy), line, font=f, fill=BLACK, anchor="mm")


def draw_credit(d: ImageDraw.ImageDraw, credit: str):
    d.text((W / 2, 42), f"TWITCH.TV/{credit.upper()}", font=font(38), fill=(255, 255, 255, 200),
           stroke_width=3, stroke_fill=(0, 0, 0, 190), anchor="mm")


def build_overlay(words, title, credit, cap_y, title_y, duration, tmp: Path) -> Path:
    """Gera 1 PNG por trecho de tempo e um arquivo concat do ffmpeg com as durações."""
    caps = caption_events(words)
    cuts = {0.0, duration}
    if title:
        cuts.add(min(TITLE_SECONDS, duration))
    for s, e, _, _ in caps:
        cuts.update((min(s, duration), min(e, duration)))
    cuts = sorted(c for c in cuts if 0 <= c <= duration)
    lines = ["ffconcat version 1.0"]
    for n, (a, b) in enumerate(zip(cuts, cuts[1:])):
        if b - a < 0.01:
            continue
        mid = (a + b) / 2
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if credit:
            draw_credit(d, credit)
        for s, e, g, hi in caps:
            if s <= mid < e:
                draw_caption(d, g, hi, cap_y)
                break
        if title and mid < TITLE_SECONDS:
            draw_title(d, title, title_y)
        p = tmp / f"o{n:04d}.png"
        img.save(p)
        lines += [f"file '{p}'", f"duration {b - a:.3f}"]
    lines.append(f"file '{p}'")  # o concat ignora a duração do último item sem repetir
    concat = tmp / "overlay.ffconcat"
    concat.write_text("\n".join(lines) + "\n")
    return concat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--cam", required=True, help="x,y,w,h em 1920x1080")
    ap.add_argument("--layout", default="cam_top", choices=["cam_top", "cam_bottom", "full"])
    ap.add_argument("--gx", type=float, default=REF_W / 2)
    ap.add_argument("--title", default="")
    ap.add_argument("--credit", default="")
    ap.add_argument("--model", default="small")
    ap.add_argument("--ia", action="store_true", help="título e gancho com Gemini Flash (GEMINI_API_KEY)")
    a = ap.parse_args()

    iw, ih, duration = probe(a.src)
    sx, sy = iw / REF_W, ih / REF_H
    cx, cy, cw, ch = (float(v) for v in a.cam.split(","))
    cx, cy, cw, ch = int(cx * sx), int(cy * sy), int(cw * sx), int(ch * sy)
    cam_x0, cam_x1 = cx, cx + cw  # caixa original da cam, para o recorte do gameplay não pegar a borda dela

    cam_h = max(480, min(760, round(W * ch / cw / 2) * 2))  # mantém a proporção da cam
    # cam muito alta (vertical) ou muito larga: recorta a caixa na proporção da área, sem esticar
    alvo = W / cam_h
    if cw / ch < alvo * 0.97:  # alta demais: corta em cima/embaixo, mirando o rosto (~45% da altura)
        nh = int(cw / alvo)
        cy = int(cy + max(0, min(ch - nh, ch * 0.45 - nh / 2)))
        ch = nh
    elif cw / ch > alvo * 1.03:  # larga demais: corta as laterais
        nw = int(ch * alvo)
        cx, cw = cx + (cw - nw) // 2, nw
    game_h = H - cam_h
    gw = int(ih * W / game_h) // 2 * 2  # recorte do gameplay com a proporção da área de baixo
    gx = int(min(max(a.gx * sx - gw / 2, 0), iw - gw))
    if a.layout != "full" and gx < cam_x1 and gx + gw > cam_x0:  # recorte encosta na cam: empurra para longe
        novo = cam_x0 - gw if (cam_x0 + cam_x1) / 2 > iw / 2 else cam_x1
        if 0 <= novo <= iw - gw:
            gx = int(novo)

    words = transcribe(a.src, a.model)
    if a.ia:  # gancho da IA vira o texto amarelo do começo; título vai para o post (lido pelo corte.py)
        import titulos
        ia = titulos.gerar(a.credit, a.title, " ".join(w["text"] for w in words), titulos.quadros(a.src, duration))
        print("ia:", ia, flush=True)
        if ia:
            a.title = ia["gancho"]
            Path(a.out).with_suffix(".ia.json").write_text(json.dumps(ia, ensure_ascii=False), encoding="utf-8")
    if a.layout == "full":  # sem facecam: gameplay em tela cheia (recorte 9:16)
        fw = int(ih * W / H) // 2 * 2
        fx = int(min(max(a.gx * sx - fw / 2, 0), iw - fw))
        cap_y, seam = 1250, 330
        base = f"[0:v]crop={fw}:{ih}:{fx}:0,scale={W}:{H},setsar=1,fps=30[base];"
        cw = ch = cx = cy = cam_h = 0
        gw, gx, game_h = fw, fx, H
    else:
        seam = cam_h if a.layout == "cam_top" else game_h
        cap_y = seam + 190 if a.layout == "cam_top" else seam - 190
        order = "[cam][g]" if a.layout == "cam_top" else "[g][cam]"
        base = (f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={W}:{cam_h},setsar=1[cam];"
                f"[0:v]crop={gw}:{ih}:{gx}:0,scale={W}:{game_h},setsar=1[g];"
                f"{order}vstack=inputs=2,fps=30[base];")
    with tempfile.TemporaryDirectory() as tmp:
        concat = build_overlay(words, a.title, a.credit, cap_y, seam, duration, Path(tmp))
        fc = (base + f"[1:v]format=rgba[ov];[base][ov]overlay=0:0:format=auto:shortest=0:eof_action=pass[v];"
              f"[0:a]loudnorm=I=-14:TP=-1.5:LRA=11[a]")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", a.src, "-f", "concat", "-safe", "0", "-i", str(concat),
                        "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-t", f"{duration:.3f}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-maxrate", "5M", "-bufsize", "10M", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", a.out], check=True)
    Path(a.out).with_suffix(".words.json").write_text(json.dumps(words, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ok {a.out} | cam {cw}x{ch}@{cx},{cy} -> {W}x{cam_h} | gameplay {gw}x{ih}@{gx} -> {W}x{game_h} | {len(words)} palavras")


if __name__ == "__main__":
    main()

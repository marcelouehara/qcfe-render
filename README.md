# qcfe-render

Monta os cortes verticais do canal **Que Corte Foi Esse** (GTA 6) a partir de clipes da Twitch.
O n8n escolhe os candidatos (clipes mais vistos de streamers que liberam corte) e dispara este repo via
`repository_dispatch` (`event_type: corte`). O GitHub Actions:

1. baixa cada candidato e descarta momentos "só câmera" (detector de rosto YuNet);
2. empilha cam do streamer + gameplay em 1080x1920, legenda palavra a palavra (faster-whisper) e título;
3. publica `<id>.mp4` e `<id>.json` no branch `videos` (GitHub Pages) e atualiza `historico.json`.

`streamers.json`: streamers liberados, posição da facecam (x,y,w,h num quadro 1920x1080) e layout.

Fonte Anton: SIL Open Font License (fonts/OFL.txt). Modelo YuNet: OpenCV Zoo (MIT).
Teste local: `JOB='{...}' python corte.py`.

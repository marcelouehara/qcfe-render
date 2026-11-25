"""Upload de Short no YouTube pela API oficial (canal QUE CORTE FOI ESSE, app marcelo-conteudo).

Credenciais nas variáveis YT_CLIENT_ID, YT_CLIENT_SECRET e YT_REFRESH_TOKEN (no GitHub: secrets do repo;
no Mac: ~/.youtube_cortes.env). Uso direto:
  python youtube.py VIDEO.mp4 "Título" "Descrição" [2026-11-21T14:00:00Z]   (data = publicar agendado)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def token(pref: str = "YT_") -> str:
    """pref = prefixo das variáveis do canal: YT_ (QUE CORTE FOI ESSE) ou YT_CONNEXION_ (@ConnexionCortesGTA6)."""
    d = urllib.parse.urlencode({"client_id": os.environ[pref + "CLIENT_ID"], "client_secret": os.environ[pref + "CLIENT_SECRET"],
                                "refresh_token": os.environ[pref + "REFRESH_TOKEN"], "grant_type": "refresh_token"}).encode()
    return json.load(urllib.request.urlopen("https://oauth2.googleapis.com/token", d, timeout=60))["access_token"]


def upload(video: str, titulo: str, descricao: str, tags: list[str] | None = None, publicar_em: str | None = None,
           pref: str = "YT_") -> dict:
    """Devolve {"status": "ok", "id", "url"} ou {"status": "erro", "motivo"} (ex.: quotaExceeded)."""
    try:
        tk = token(pref)
        status = {"privacyStatus": "private", "publishAt": publicar_em} if publicar_em else {"privacyStatus": "public"}
        status.update({"selfDeclaredMadeForKids": False, "containsSyntheticMedia": False})
        meta = {"snippet": {"title": titulo[:100], "description": descricao[:4900], "tags": tags or [],
                            "categoryId": "20", "defaultLanguage": "pt-BR", "defaultAudioLanguage": "pt-BR"},
                "status": status}
        size = os.path.getsize(video)
        req = urllib.request.Request(
            "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
            data=json.dumps(meta).encode(), method="POST",
            headers={"Authorization": "Bearer " + tk, "Content-Type": "application/json; charset=UTF-8",
                     "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)})
        loc = urllib.request.urlopen(req, timeout=60).headers["Location"]
        with open(video, "rb") as f:
            put = urllib.request.Request(loc, data=f.read(), method="PUT",
                                         headers={"Content-Type": "video/mp4", "Content-Length": str(size)})
            v = json.load(urllib.request.urlopen(put, timeout=600))
        return {"status": "ok", "id": v["id"], "url": "https://www.youtube.com/shorts/" + v["id"],
                "publicar_em": publicar_em}
    except urllib.error.HTTPError as e:
        corpo = e.read().decode()[:500]
        motivo = "quotaExceeded" if "quota" in corpo.lower() else f"HTTP {e.code}: {corpo}"
        return {"status": "erro", "motivo": motivo}
    except Exception as e:  # noqa: BLE001 - qualquer falha vira reserva pelo Buffer no n8n
        return {"status": "erro", "motivo": repr(e)[:300]}


if __name__ == "__main__":
    print(json.dumps(upload(sys.argv[1], sys.argv[2], sys.argv[3], ["gta 6", "gta vi", "gta6", "cortes"],
                            sys.argv[4] if len(sys.argv) > 4 else None), ensure_ascii=False))

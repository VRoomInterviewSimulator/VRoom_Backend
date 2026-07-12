"""
TTS 워커(Node B) 연동. 면접관 대사 텍스트를 Node B 웹소켓으로 보내고,
돌아오는 44.1kHz / Mono / 32-bit Float PCM 오디오 스트림을 청크 단위로 흘려준다.

저지연을 위해 대사를 구(phrase) 단위로 잘라 순차 합성한다 = 스트리밍 처리.
"""
from __future__ import annotations

import re
import json
from typing import AsyncIterator

# 문장부호 기준으로 구를 나눠 TTS에 먼저 들어가는 구부터 빠르게 합성.
_SPLIT = re.compile(r"(?<=[.?!。…\n])\s+|(?<=[,，])\s+")


def split_phrases(text: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT.split(text) if p.strip()]
    return parts or [text]


async def synthesize_ws_stream(tts_ws, dialogue: str) -> AsyncIterator[bytes | str]:
    """영속 웹소켓(tts_ws)을 통해 대사를 통째로 Node B에 보내고 PCM 청크 또는 자막 JSON을 yield."""
    # 대사를 split_phrases로 분할하지 않고 한 번에 전송하여 전체 문맥(어조) 유지
    await tts_ws.send(json.dumps({"text": dialogue}))
    while True:
        m = await tts_ws.recv()
        if isinstance(m, str):
            try:
                event = json.loads(m)
                if event.get("type") == "end":
                    break
                elif event.get("type") == "subtitle":
                    yield m  # 자막 JSON 문자열 그대로 yield
            except json.JSONDecodeError:
                pass
        else:
            yield m
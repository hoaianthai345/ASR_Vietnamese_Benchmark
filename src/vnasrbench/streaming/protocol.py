from __future__ import annotations

from dataclasses import dataclass

from vnasrbench.streaming.chunker import ChunkConfig


@dataclass(frozen=True)
class StreamingConfig:
    enabled: bool
    chunk_ms: int
    overlap_ms: int
    lookahead_ms: int
    sample_rate: int = 16000

    def to_chunk_config(self) -> ChunkConfig:
        return ChunkConfig(
            sample_rate=self.sample_rate,
            chunk_ms=self.chunk_ms,
            overlap_ms=self.overlap_ms,
            lookahead_ms=self.lookahead_ms,
        )

"""
speechlens/models.py
Pydantic v2 models mirroring schema/analysis_result.schema.json (DESIGN §12).
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class SoftwareMeta(BaseModel):
    speechlens: str
    model: str


class Meta(BaseModel):
    speech_id: Optional[str]
    mode: str  # "reference" | "norm"
    duration_s: float
    n_words: int
    baseline_readers: list[str]
    software: SoftwareMeta
    input_sha256: str


class Dimensions(BaseModel):
    pacing: float
    intonation: float
    fluency: float
    volume: float
    clarity: float


class Scores(BaseModel):
    total: float
    dimensions: Dimensions


class GlobalStats(BaseModel):
    articulation_rate_syl_s: float
    baseline_articulation_rate_syl_s: Optional[float]
    wpm: float
    pause_ratio: float
    f0_sd_st: float


class ZScores(BaseModel):
    tempo: Optional[float] = None
    pitch: Optional[float] = None
    energy: Optional[float] = None
    pause: Optional[float] = None


class Prominence(BaseModel):
    user: Optional[float] = None
    base: Optional[float] = None


class WordResult(BaseModel):
    i: int
    text: str
    start: float
    end: float
    base_start: Optional[float]
    base_end: Optional[float]
    z: ZScores
    prominence: Prominence
    align_score: float
    flags: list[str]


class SeriesBand(BaseModel):
    user: list[Optional[float]]
    base: list[Optional[float]]
    lo: list[Optional[float]]
    hi: list[Optional[float]]


class TempoSeries(BaseModel):
    user: list[Optional[float]]
    base: list[Optional[float]]


class Series(BaseModel):
    hop_s: float
    t_user: list[float]
    t_base: list[float]
    pitch_st: SeriesBand
    energy_db: SeriesBand
    tempo: TempoSeries


class Pause(BaseModel):
    start: float
    end: float
    kind: str  # "silent" | "filled" | "other"
    expected: bool


class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    metric: Optional[str] = None
    unit: Optional[str] = None
    observed: Optional[float] = None
    expected: Optional[float] = None
    sigma: Optional[float] = None
    z: Optional[float] = None
    formula: Optional[str] = None


class Explanation(BaseModel):
    summary: str
    context: str
    effect: str
    fix: str
    math: str


class Region(BaseModel):
    id: str
    family: str
    dimension: str
    start: float
    end: float
    word_idx: list[int] = Field(..., min_length=2, max_length=2)
    severity: float
    z_peak: float
    cost: float
    evidence: Evidence
    explanation: Explanation


class AnalysisResult(BaseModel):
    schema_version: str = "1.0"
    meta: Meta
    global_: GlobalStats = Field(alias="global")
    scores: Scores
    words: list[WordResult]
    series: Series
    pauses: list[Pause]
    regions: list[Region]
    warnings: list[str]

    model_config = ConfigDict(populate_by_name=True)

    def model_dump_api(self) -> dict:
        """Dump with `global` key (not `global_`) for API responses."""
        return self.model_dump(by_alias=True)
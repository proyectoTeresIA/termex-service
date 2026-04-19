"""Main FastAPI application."""
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI

from api import config
from api.models import (
    BatchExtractRequest,
    BatchJobResponse,
    ExtractRequest,
    ExtractResponse,
)


@lru_cache
def get_attention_extractor():
    from api.extractors import AttentionRankExtractor
    return AttentionRankExtractor()


@lru_cache
def get_mde_extractor():
    from api.extractors.mderank_client import MDERankExtractor
    return MDERankExtractor()


app = FastAPI(
    title="TermEx API",
    description="API REST para extracción de términos con AttentionRank y MDERank",
    version="1.0.0",
)


@app.post("/attentionrank", response_model=ExtractResponse)
def extract_attentionrank(request: ExtractRequest) -> ExtractResponse:
    """Extrae términos usando AttentionRank."""
    extractor = get_attention_extractor()
    terms = extractor.extract(request.doc, request.k_val)
    return ExtractResponse(terms=terms)


@app.post("/mderank", response_model=ExtractResponse)
def extract_mderank(request: ExtractRequest) -> ExtractResponse:
    """Extrae términos usando MDERank."""
    extractor = get_mde_extractor()
    terms = extractor.extract(request.doc, request.k_val)
    return ExtractResponse(terms=terms)


@app.post("/attentionrank/batch", response_model=BatchJobResponse)
def batch_attentionrank(request: BatchExtractRequest) -> BatchJobResponse:
    """Inicio de trabajo por lotes con AttentionRank (futuro)."""
    import uuid
    work_id = str(uuid.uuid4())
    return BatchJobResponse(work_id=work_id)


@app.post("/mderank/batch", response_model=BatchJobResponse)
def batch_mderank(request: BatchExtractRequest) -> BatchJobResponse:
    """Inicio de trabajo por lotes con MDERank (futuro)."""
    import uuid
    work_id = str(uuid.uuid4())
    return BatchJobResponse(work_id=work_id)

"""Pydantic models para la API."""
from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    """Request para extracción de un solo documento."""
    doc: str = Field(..., description="Texto del documento")
    k_val: int = Field(..., ge=1, description="Número de términos a devolver")


class ExtractResponse(BaseModel):
    """Response con los términos extraídos."""
    terms: list[str] = Field(..., description="Lista de términos extraídos ordenados")


class BatchExtractRequest(BaseModel):
    """Request para extracción de múltiples documentos (futuro)."""
    docs: list[str] = Field(..., description="Lista de textos de documentos")
    k_val: int = Field(..., ge=1, description="Número de términos a devolver por documento")


class BatchJobResponse(BaseModel):
    """Response inicial para trabajos por lotes (futuro)."""
    work_id: str = Field(..., description="ID del trabajo para recuperar resultados")


class JobStatusResponse(BaseModel):
    """Status de un trabajo por lotes (futuro)."""
    work_id: str
    status: str
    results: list[list[str]] | None = None

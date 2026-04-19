"""Tests para la API."""
import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport
from api.main import app


@pytest.fixture(autouse=True)
def mock_extractors():
    mock_attention = MagicMock()
    mock_attention.extract.return_value = ["artificial", "inteligencia", "tecnología", "mundo", "transformando"]
    
    mock_mde = MagicMock()
    mock_mde.extract.return_value = ["inteligencia", "artificial", "transformando", "tecnología", "mundo"]
    
    with patch("api.main.get_attention_extractor", return_value=mock_attention), \
         patch("api.main.get_mde_extractor", return_value=mock_mde):
        yield


@pytest.mark.asyncio
async def test_attentionrank_returns_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/attentionrank",
            json={"doc": "La inteligencia artificial está transformando el mundo.", "k_val": 5}
        )
    assert response.status_code == 200
    data = response.json()
    assert "terms" in data
    assert isinstance(data["terms"], list)


@pytest.mark.asyncio
async def test_mderank_returns_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/mderank",
            json={"doc": "La inteligencia artificial está transformando el mundo.", "k_val": 5}
        )
    assert response.status_code == 200
    data = response.json()
    assert "terms" in data
    assert isinstance(data["terms"], list)


@pytest.mark.asyncio
async def test_attentionrank_calls_extractor_with_k_val():
    from unittest.mock import patch, MagicMock
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = ["term1", "term2", "term3"]
    
    with patch("api.main.get_attention_extractor", return_value=mock_extractor):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/attentionrank",
                json={"doc": "test document", "k_val": 3}
            )
    
    assert response.status_code == 200
    mock_extractor.extract.assert_called_once_with("test document", 3)
    data = response.json()
    assert data["terms"] == ["term1", "term2", "term3"]


@pytest.mark.asyncio
async def test_batch_endpoints_return_work_id():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for endpoint in ["/attentionrank/batch", "/mderank/batch"]:
            response = await client.post(
                endpoint,
                json={"docs": ["doc1", "doc2"], "k_val": 5}
            )
            assert response.status_code == 200
            data = response.json()
            assert "work_id" in data


@pytest.mark.asyncio
async def test_invalid_k_val():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/attentionrank",
            json={"doc": "test", "k_val": 0}
        )
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_missing_doc_field():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/attentionrank",
            json={"k_val": 5}
        )
        assert response.status_code == 422

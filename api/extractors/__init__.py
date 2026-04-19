"""Cliente para AttentionRank."""
import os
import shutil
import tempfile
from pathlib import Path

from attentionrank import AttentionRank, AttentionRankConfig
from api import config


_patched = False


class AttentionRankExtractor:
    """Wrapper para AttentionRank."""

    def __init__(
        self,
        model_path: str = None,
        lang: str = None,
    ):
        global _patched
        import attentionrank
        from attentionrank import attentionrank as ar_module
        
        self._pkg_dir = Path(attentionrank.__file__).parent
        self._orig_cwd = Path.cwd()
        self._model_path = str(Path(model_path or config.MODEL_PATH).expanduser().resolve())
        self._lang = lang or config.MODEL_LANG
        self._stopwords = {}
        for f in self._pkg_dir.glob("*.txt"):
            self._stopwords[f.name] = f.read_text()
        
        if not _patched:
            original_read = ar_module.read_term_list_file
            stopwords_cache = self._stopwords
            
            def patched_read(filepath):
                filename = os.path.basename(filepath)
                if filename in stopwords_cache:
                    return [l.strip() for l in stopwords_cache[filename].splitlines() if l.strip()]
                try:
                    return original_read(filepath)
                except FileNotFoundError:
                    if filename in stopwords_cache:
                        return [l.strip() for l in stopwords_cache[filename].splitlines() if l.strip()]
                    raise
            
            ar_module.read_term_list_file = patched_read
            _patched = True
            
            ar_module.read_term_list_file = patched_read
            _patched = True

    def extract(self, doc: str, k_val: int) -> list[str]:
        """Extrae términos de un documento."""
        work_dir = Path(tempfile.mkdtemp())
        pkg_attentionrank = self._pkg_dir / "attentionrank"
        workdir_attentionrank = work_dir / "attentionrank"
        os.makedirs(workdir_attentionrank, exist_ok=True)
        for f in pkg_attentionrank.glob("*.txt"):
            shutil.copy2(f, workdir_attentionrank)
        
        doc_path = work_dir / "docsutf8"
        doc_path.mkdir(exist_ok=True)
        (doc_path / "doc_0.txt").write_text(doc, encoding="utf-8")
        
        config = AttentionRankConfig(
            lang=self._lang,
            model_name_or_path=self._model_path,
            model_type="roberta",
            log_level="WARNING",
        )
        extractor = AttentionRank(config)
        
        orig = os.getcwd()
        os.chdir(work_dir)
        try:
            results = extractor.extract(".", k_val)
        finally:
            os.chdir(orig)
        
        shutil.rmtree(work_dir, ignore_errors=True)
        return results[0] if results else []

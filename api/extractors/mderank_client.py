"""Cliente para MDERank."""
import tempfile
import os
from pathlib import Path

from mderank import MDERank, MDERankConfig
from api import config


def patched_evaluate(self, key_dir, labels, labels_stemed, k):
    if not labels:
        return
    num_c = 0
    num_e = 0
    num_s = 0
    files = sorted(os.listdir(key_dir))
    for i, fname in enumerate(files):
        if i >= len(labels):
            break
        path = os.path.join(key_dir, fname)
        with open(path, encoding="utf-8") as f:
            preds = [l.strip().lower() for l in f.readlines()][:k]
        j = 0
        for temp in preds:
            tokens = temp.split()
            tt = ' '.join(self.porter.stem(t) for t in tokens)
            if tt in (labels_stemed[i] if i < len(labels_stemed) else []) or temp in labels[i]:
                if j < k:
                    num_c += 1
            j += 1
        num_e += len(preds)
        num_s += len(labels[i])


_patched = False


class MDERankExtractor:
    """Wrapper para MDERank."""

    def __init__(
        self,
        model_path: str = None,
        lang: str = None,
    ):
        global _patched
        from mderank.mderank_core import MDERankModel
        
        if not _patched:
            MDERankModel.evaluate_from_files = patched_evaluate
            _patched = True
        
        cfg = MDERankConfig(
            lang=lang or config.MODEL_LANG,
            model_name_or_path=model_path or config.MODEL_PATH,
            model_type="roberta",
            log_level="WARNING",
        )
        self._extractor = MDERank(cfg)
        self._work_dir = Path(tempfile.mkdtemp())
        self._output_dir = self._work_dir / "output"
        self._output_dir.mkdir(exist_ok=True)

    def extract(self, doc: str, k_val: int) -> list[str]:
        """Extrae términos de un documento."""
        doc_path = self._work_dir / "docsutf8"
        doc_path.mkdir(exist_ok=True)
        doc_file = doc_path / "doc_0.txt"
        doc_file.write_text(doc, encoding="utf-8")
        
        for subdir in ["top5", "top10", "top15"]:
            os.makedirs(self._output_dir / subdir, exist_ok=True)
        
        try:
            results = self._extractor.extract(str(self._work_dir), k_val)
            return results[0] if results else []
        finally:
            for f in doc_path.glob("*.txt"):
                f.unlink()

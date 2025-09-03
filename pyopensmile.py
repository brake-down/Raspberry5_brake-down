import opensmile as _opensmile

# ---- re-export 핵심 API (이 이름들을 스크립트가 기대함) ----
Smile = _opensmile.Smile
FeatureSet = _opensmile.FeatureSet
FeatureLevel = _opensmile.FeatureLevel

# 선택: 버전도 그대로 노출
__version__ = getattr(_opensmile, "__version__", None)

# 무엇이 공개되는지 명시
__all__ = ["Smile", "FeatureSet", "FeatureLevel", "__version__"]

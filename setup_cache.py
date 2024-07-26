from transformers import Cache

def setup_cache(self, cache_implementation: str, max_batch_size: int, max_cache_len: int, model_kwargs, cache_config=None) -> Cache:

    cache_implementation = self.config.get("cache_implementation") or cache_implementation
    cache_cls = CACHE_CLASSES_MAPPING[cache_implementation]

    requires_cross_attention_cache = (
        self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
    )

    if hasattr(self, "_cache"):
        cache_to_check = self._cache.self_attention_cache if requires_cross_attention_cache else self._cache

    max_cache_len = min(self.config.get("sliding_window", math.inf), max_cache_len)
    need_new_cache = self._needs_new_cache(cache_to_check, model_kwargs, requires_cross_attention_cache)

    if need_new_cache:
        if hasattr(self.config, "_pre_quantization_dtype"):
            cache_dtype = self.config._pre_quantization_dtype
        else:
            cache_dtype = self.dtype
        cache_kwargs = {
            "config": self.config,
            "max_batch_size": max_batch_size,
            "max_cache_len": max_cache_len,
            "device": self.device,
            "dtype": cache_dtype,
        }
        self._cache = cache_cls(**cache_kwargs)
        if requires_cross_attention_cache:
            encoder_kwargs = cache_kwargs.copy()
            encoder_kwargs["max_cache_len"] = model_kwargs["encoder_outputs"][0].shape[1]
            self._cache = EncoderDecoderCache(self._cache, cache_cls(**encoder_kwargs))
    else:
        self._cache.reset()

    return self._cache

def _needs_new_cache(self, cache_to_check, model_kwargs, requires_cross_attention_cache):
    need_new_cache = (
        not hasattr(self, "_cache")
        or (not isinstance(cache_to_check, cache_cls))
        or cache_to_check.max_batch_size != max_batch_size
    )

    need_new_cache = need_new_cache or cache_to_check.get("max_cache_len", math.inf) < max_cache_len

    if requires_cross_attention_cache and hasattr(self, "_cache"):
        need_new_cache = (
            need_new_cache
            or self._cache.cross_attention_cache.max_cache_len != model_kwargs["encoder_outputs"][0].shape[1]
        )
    
    return need_new_cache
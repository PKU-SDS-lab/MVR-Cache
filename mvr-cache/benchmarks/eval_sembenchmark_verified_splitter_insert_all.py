"""
Insert-all variant of `eval_sembenchmark_verified_splitter.py`.

This keeps the original CLI and evaluation logic, but changes the cache update rule:
every processed sample is inserted into the cache/index regardless of whether the
request was a cache hit or a cache miss.

Implementation strategy:
- Reuse the original benchmark script unchanged via import.
- Monkey-patch its `VerifiedSplitterDecisionPolicy` symbol with an insert-all subclass.
- Keep the original Bayesian update behavior for the retrieved neighbor metadata.
- Avoid duplicate insertions by disabling the base class's miss-only insert inside the
  background callback path.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

# Ensure imports resolve to the local repo when this file is run directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # .../vcache_Multi_HNSW
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import benchmarks.eval_sembenchmark_verified_splitter as base_eval
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_policy.strategies.verified import _Action


class InsertAllVerifiedSplitterDecisionPolicy(base_eval.VerifiedSplitterDecisionPolicy):
    """
    Same splitter-based verified policy, except every request is inserted into cache.

    Insert behavior:
    - cold start/no candidate: add current sample (same as base)
    - exploit/hit: also add the current sample
    - explore/miss: add the current sample immediately

    The Bayesian observation update for the retrieved NN still happens asynchronously,
    but we suppress the base policy's conditional miss-only insert in the callback
    because the current sample was already inserted at request time.
    """

    def process_request(
        self, prompt: str, system_prompt: str | None, id_set: int
    ) -> Tuple[bool, str, EmbeddingMetadataObj]:
        if self.inference_engine is None or self.cache is None:
            raise ValueError("Policy has not been setup")

        if self.splitter is None:
            raise ValueError(
                "VerifiedSplitterDecisionPolicy requires `splitter` (MaxSimSplitter) to be provided."
            )

        # Keep the same optimized query encoding + retrieval path as the base policy.
        with self._time_block("splitter.encode_text"):
            query_enc = self.splitter.encode_text(prompt)
        query_knn_emb = query_enc["pooled_knn"]
        query_knn_emb_cpu = query_knn_emb.detach().float().cpu().tolist()

        nn_metadata, similarity_score = self._select_nn_by_maxsim_with_query(
            prompt, query_enc, query_knn_emb_cpu
        )

        if nn_metadata is None or similarity_score is None:
            response = self.inference_engine.create(prompt=prompt, system_prompt=system_prompt)
            self._VerifiedSplitterDecisionPolicy__cache_add(
                prompt=prompt, response=response, id_set=id_set
            )
            return False, response, EmbeddingMetadataObj(embedding_id=-1, response="")

        action = self.bayesian.select_action(
            similarity_score=similarity_score, metadata=nn_metadata
        )

        match action:
            case _Action.EXPLOIT:
                # Even on hits, insert the current sample so every request is indexed.
                current_response = self.inference_engine.create(
                    prompt=prompt, system_prompt=system_prompt
                )
                self._VerifiedSplitterDecisionPolicy__cache_add(
                    prompt=prompt, response=current_response, id_set=id_set
                )
                return True, nn_metadata.response, nn_metadata

            case _Action.EXPLORE:
                response = self.inference_engine.create(prompt=prompt, system_prompt=system_prompt)
                self._VerifiedSplitterDecisionPolicy__update_cache(
                    response=response,
                    nn_metadata=nn_metadata,
                    similarity_score=similarity_score,
                    embedding_id=nn_metadata.embedding_id,
                    prompt=prompt,
                    label_id_set=id_set,
                )
                self._VerifiedSplitterDecisionPolicy__cache_add(
                    prompt=prompt, response=response, id_set=id_set
                )
                return False, response, nn_metadata

        raise RuntimeError("Unexpected action returned by Bayesian policy")

    # Override the *base-class private* callback target by using the mangled name.
    # Base setup() resolves `self.__perform_cache_update` to this attribute lookup.
    def _VerifiedSplitterDecisionPolicy__perform_cache_update(self, update_args: tuple) -> None:
        (
            should_have_exploited,
            _new_response,
            similarity_score,
            embedding_id,
            _prompt,
            _id_set,
        ) = update_args

        if self.cache is None:
            return

        try:
            latest_metadata_object = self.cache.get_metadata(embedding_id=embedding_id)
        except (ValueError, KeyError):
            return

        if latest_metadata_object is None:
            return

        try:
            self.bayesian.add_observation_to_metadata(
                similarity_score=similarity_score,
                is_correct=should_have_exploited,
                metadata=latest_metadata_object,
            )
        except (ValueError, KeyError):
            return

        try:
            self.cache.update_metadata(
                embedding_id=embedding_id, embedding_metadata=latest_metadata_object
            )
        except (ValueError, KeyError):
            return


def main() -> None:
    # Swap in the insert-all policy while reusing the original benchmark implementation.
    base_eval.VerifiedSplitterDecisionPolicy = InsertAllVerifiedSplitterDecisionPolicy
    base_eval.main()


if __name__ == "__main__":
    main()

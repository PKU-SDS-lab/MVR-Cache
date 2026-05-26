"""
Insert-all variant of `eval_sembenchmark_verified.py`.

This keeps the original CLI and benchmark output behavior, but changes the verified
policy so every processed sample is inserted into cache regardless of hit or miss.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # .../vcache_Multi_HNSW
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import benchmarks.eval_sembenchmark_verified as base_eval
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_policy.strategies.verified import _Action


class InsertAllVerifiedDecisionPolicy(base_eval.VerifiedDecisionPolicy):
    """
    Same verified policy, except every request is inserted into cache.

    - cold start/no neighbor: same as base behavior
    - exploit/hit: insert current sample as well
    - explore/miss: insert current sample immediately

    The base callback already updates Bayesian observations for the retrieved neighbor.
    We override its final cache-update callback to avoid a second miss-only insert.
    """

    def process_request(
        self, prompt: str, system_prompt: str | None, id_set: int
    ) -> Tuple[bool, str, EmbeddingMetadataObj]:
        if self.inference_engine is None or self.cache is None:
            raise ValueError("Policy has not been setup")

        with self._time_block("embedding"):
            emb = self.cache.embedding_engine.get_embedding(prompt)
        with self._time_block("retrieval"):
            knn = self.cache.embedding_store.get_knn(emb, k=1)

        if not knn:
            response = self.inference_engine.create(prompt=prompt, system_prompt=system_prompt)
            self.cache.add(prompt=prompt, response=response, id_set=id_set)
            return False, response, EmbeddingMetadataObj(embedding_id=-1, response="")

        similarity_score, embedding_id = knn[0]

        try:
            nn_metadata: EmbeddingMetadataObj = self.cache.get_metadata(
                embedding_id=embedding_id
            )
        except Exception:
            new_response: str = self.inference_engine.create(
                prompt=prompt, system_prompt=system_prompt
            )
            self.cache.add(prompt=prompt, response=new_response, id_set=id_set)
            return (
                False,
                new_response,
                EmbeddingMetadataObj(embedding_id=-1, response=""),
            )

        action = self.bayesian.select_action(
            similarity_score=similarity_score, metadata=nn_metadata
        )

        match action:
            case _Action.EXPLOIT:
                current_response = self.inference_engine.create(
                    prompt=prompt, system_prompt=system_prompt
                )
                self.cache.add(prompt=prompt, response=current_response, id_set=id_set)
                return True, nn_metadata.response, nn_metadata

            case _Action.EXPLORE:
                response = self.inference_engine.create(
                    prompt=prompt, system_prompt=system_prompt
                )
                self._VerifiedDecisionPolicy__update_cache(
                    response=response,
                    nn_metadata=nn_metadata,
                    similarity_score=similarity_score,
                    embedding_id=embedding_id,
                    prompt=prompt,
                    label_id_set=id_set,
                )
                self.cache.add(prompt=prompt, response=response, id_set=id_set)
                return False, response, nn_metadata

        raise RuntimeError("Unexpected action returned by Bayesian policy")

    def _VerifiedDecisionPolicy__perform_cache_update(self, update_args: tuple) -> None:
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
            latest_metdata_object = self.cache.get_metadata(embedding_id=embedding_id)
        except (ValueError, KeyError):
            return

        if latest_metdata_object is None:
            return

        try:
            self.bayesian.add_observation_to_metadata(
                similarity_score=similarity_score,
                is_correct=should_have_exploited,
                metadata=latest_metdata_object,
            )
        except (ValueError, KeyError):
            return

        try:
            self.cache.update_metadata(
                embedding_id=embedding_id, embedding_metadata=latest_metdata_object
            )
        except (ValueError, KeyError):
            return


def main() -> None:
    base_eval.VerifiedDecisionPolicy = InsertAllVerifiedDecisionPolicy
    base_eval.main()


if __name__ == "__main__":
    main()

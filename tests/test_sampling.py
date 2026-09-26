import numpy as np

from sampling import build_negative_pools, candidates_for_k, split_communities


def test_candidate_sets_are_nested_and_permuted_per_k():
    pool = ["b", "c", "d", "e"]
    k2 = candidates_for_k("a", pool, 2, 42)
    k4 = candidates_for_k("a", pool, 4, 42)

    assert set(k2) < set(k4)
    assert k2 == candidates_for_k("a", pool, 2, 42)
    assert k4 == candidates_for_k("a", pool, 4, 42)


def test_splits_and_negative_mining_stay_within_track():
    community_ids = [f"c{index}" for index in range(8)]
    groups = split_communities(community_ids, unseen_fraction=0.25, max_k=2, seed=9)
    communities = [
        {"community_id": community_id, "split_group": groups[community_id]}
        for community_id in community_ids
    ]
    vectors = {
        community_id: np.array([1.0, index + 1.0], dtype=np.float32)
        for index, community_id in enumerate(community_ids)
    }

    pools = build_negative_pools(communities, vectors, semantic_pool_size=3, seed=9)

    for pool in pools:
        assert pool["target"] not in pool["negative_pool"]
        assert all(groups[candidate] == pool["community_track"] for candidate in pool["negative_pool"])
        if pool["difficulty"] == "hard":
            expected = "hard_content" if pool["community_track"] == "seen" else "hard_description"
            assert pool["mining_strategy"] == expected

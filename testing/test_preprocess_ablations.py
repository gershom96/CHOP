import json

from datasets.preprocess_scand_a_chop import _process_annotation_file


def test_ablation_target_modes_preserve_candidate_identity(tmp_path):
    images = tmp_path / "images"
    bag = images / "bag"
    bag.mkdir(parents=True)
    (bag / "img_7.png").write_bytes(b"image")
    paths = {str(i): {"points": [[0, 0, 0], [float(i), 0, 0]]} for i in range(4)}
    annotation = {"bag": "bag", "annotations_by_stamp": {"7": {
        "frame_idx": 1, "paths": paths, "preference": ["2", "3", "0", "1"],
        "pairwise": [{"pair": [2, 3], "choice": 2}],
    }}}
    source = tmp_path / "annotation.json"
    source.write_text(json.dumps(annotation))

    preferred = _process_annotation_file(source, images, "png", 2, "preferred")
    original = _process_annotation_file(source, images, "png", 2, "original")
    guided = _process_annotation_file(source, images, "png", 2, "human_guided")
    all_candidates = _process_annotation_file(source, images, "png", 2, "all")

    assert preferred[0]["target_id"] == "2"
    assert original[0]["target_id"] == "0"
    assert guided[0]["target_id"] == "3"
    assert {row["target_id"] for row in all_candidates} == {"0", "1", "2", "3"}

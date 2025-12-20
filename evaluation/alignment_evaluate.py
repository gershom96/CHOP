from tqdm import tqdm
import numpy as np

from evaluation.base_evaluator import BaseEvaluator

class AlignmentEvaluator(BaseEvaluator):
    def __init__(self, output_path: str, scand: bool = True, model: str = "omnivla"):
        super().__init__(output_path, model)

        self._eval_name = "alignment"
        if scand:
            self._eval_name += "_scand"

        self._open_output_files()

    def analyze_bag(self, finetuned: bool = True):
        print(f"[INFO] Analyzing path alignment for {self.bag_name}")

        alignments = []

        for frame in tqdm(self.frames, desc=f"Eval {self._eval_name} {self.bag_name}", unit="frame"):
            if frame.goal_idx == -1:
                continue

            gt_path = frame.path_gt
            pred_path = frame.path_ft if finetuned else frame.path_bl
            if gt_path is None or pred_path is None:
                continue

            gt_path = np.asarray(gt_path)
            pred_path = np.asarray(pred_path)
            n = min(len(gt_path), len(pred_path))
            if n == 0:
                continue

            alignment = float(np.linalg.norm(gt_path[:n, :2] - pred_path[:n, :2], axis=1).mean())
            alignments.append(alignment)

        return alignments

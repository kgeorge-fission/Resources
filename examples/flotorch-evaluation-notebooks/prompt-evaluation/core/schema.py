from typing import List, Dict, Union
from flotorch_eval.llm_eval import EvaluationItem

EvaluationDatasetType = List[Dict[str, Union[str, List[EvaluationItem]]]]
PromptEvaluationResult = List[Dict[str, Union[str, Dict[str, float]]]]

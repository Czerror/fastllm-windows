from .protocal.openai_protocol import ModelCard, ModelList, ModelPermission
import time


class FastLLmModel:
    def __init__(self,
                 model_name,
                 ):
        self.model_name = model_name
        self.created_time = int(time.time())
        
        # 使用 Pydantic 模型构建响应
        model_card = ModelCard(
            id=model_name,
            object="model",
            created=self.created_time,
            owned_by="fastllm",
            root=model_name,
            parent=None,
            permission=[
                ModelPermission(
                    allow_create_engine=False,
                    allow_sampling=True,
                    allow_logprobs=True,
                    allow_search_indices=False,
                    allow_view=True,
                    allow_fine_tuning=False,
                    organization="*",
                    group=None,
                    is_blocking=False
                )
            ]
        )
        
        model_list = ModelList(
            object="list",
            data=[model_card]
        )
        
        self.response = model_list.model_dump()
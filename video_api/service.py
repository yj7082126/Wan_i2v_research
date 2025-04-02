from abc import abstractmethod
from typing import Protocol, TypeVar

ConfigType = TypeVar("ConfigType", contravariant=True)
InputType = TypeVar("InputType", contravariant=True)
OutputType = TypeVar("OutputType", covariant=True)

class InferenceService(Protocol[ConfigType, InputType, OutputType]):
    """
    Stable Diffusion Model 의 inference 기능 추상화
    본 인터페이스를 구현하는 클래스는, __init__, __call__, __del__ method 를 구현해야함.
    본 인터페이스는 다음과 같이 사용됨
    eg)
        # 로드시에 service 생성
        config = ConfigType()
        service = InferenceService(config)
        # 서비스 (다회) 사용
        inputs: List[InputType] = [input0, input1, input2]
        result: OutputType = service(input[0])
        result = service(inputs[1])
        result = service(inputs[2])
        ...
        # 언로드시에 service 명시적으로 삭제
        del service
    """

    @abstractmethod
    def __init__(self, config: ConfigType) -> None:
        """
        Inference service 로드시 호출되는 생성자.
        모델 로드시에 호출되어야하는 함수는 생성자에 모두 정의되어야함.
        """
        ...

    def __call__(self, inp: InputType) -> OutputType:
        """
        서비스 inference 실행. 실행된 서비스는 이미지 생성, 저장후 output 반환
        """
        raise NotImplementedError("__call__ method must be implemented")

    def __del__(self):
        """
        서비스 언로드시 호출되는 함수.
        """
        ...
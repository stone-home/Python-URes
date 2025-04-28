from abc import abstractmethod, ABC


class AbcAlgorithm(ABC):
    @abstractmethod
    def malloc(self, size: int, **kwargs):
        """
        Allocate memory on the device.
        """
        pass

    @abstractmethod
    def free(self, ptr: str, **kwargs):
        """
        Free memory on the device.
        """
        pass

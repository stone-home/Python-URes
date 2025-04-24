from abc import abstractmethod, ABC


class AllocatorInterface(ABC):
    def malloc(self, args, **kwargs):
        """
        Allocate memory on the device.
        """
        pass

    def free(self, args, **kwargs):
        """
        Free memory on the device.
        """
        pass
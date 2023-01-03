class Queue:
    """
    Specific implementation of queue respresenting Tabu List
    used in TabuSearch Algorithm

    Queue is iterable and elements can be accesed with indices

    Methods:
        enqueuqe:
            add an element to the end of the queue
            if limit of elements was exceeded exception is raised
        dequeue:
            removes an element from the beginning of the queue
            works only if queue is full
    """

    def __init__(self, length: int = 3) -> None:
        """
        Params:
            length: int
                Max length of the queue
        """
        self._limit = length
        self._queue = []
        self.length = length

    def enqueue(self, object_: object):
        """
        Adds object to the end of the queue

        Raises:
            AssertionError
                if queue is full attempt will raise an exception
        """
        assert len(self) < self._limit, "Queue overflow!"
        assert isinstance(object_, tuple), "object must be a tuple!"
        self._queue.append(object_)

    def dequeue(self):
        """
        Removes object from the beginning of the queue

        If queue is not full, first element will not be removed
        """
        if len(self) == self._limit:
            self._queue.pop(0)

    def __len__(self):
        return len(self._queue)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [self[i] for i in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"The index {key} is out of range")
            return self._queue[key]
        else:
            raise TypeError("Invalid argument type")

    def __iter__(self):
        self.__i = 0
        return self

    def __next__(self):
        if self.__i >= len(self._queue):
            raise StopIteration
        element = self._queue[self.__i]
        self.__i += 1
        return element

    def __str__(self) -> str:
        return str(list(self._queue))

    def __repr__(self) -> str:
        return str(self)

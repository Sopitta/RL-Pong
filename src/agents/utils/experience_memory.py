"""
code from https://github.com/PacktPublishing/Hands-On-Intelligent-Agents-with-OpenAI-Gym/blob/master/ch6/utils/experience_memory.py

"""
from collections import namedtuple
import random

Experience = namedtuple("Experience", ['obs','action','next_obs', 'reward', 'done'])


class ExperienceMemory(object):
    """
    A cyclic/ring buffer based Experience Memory implementation
    """
    def __init__(self, capacity=int(1e4)):
        """
        :param capacity: Total capacity (Max number of Experiences)
        :return:
        """
        self.capacity = capacity
        self.mem_idx = 0  # Index of the current experience
        self.memory = []

    def store(self, experience):
        """
        :param experience: The Experience object to be stored into the memory
        :return:
        """
        if self.mem_idx < self.capacity:
            # Extend the memory and create space
            self.memory.append(None)
        self.memory[self.mem_idx % self.capacity] = experience
        self.mem_idx += 1

    def sample(self, batch_size):
        """
        :param batch_size:  Sample batch_size
        :return: A list of batch_size number of Experiences sampled at random from mem
        """
        assert batch_size <= len(self.memory), "Sample batch_size is more than available exp in mem"
        return random.sample(self.memory, batch_size)

    def get_size(self):
        """
        :return: Number of Experiences stored in the memory
        """
        return len(self.memory)


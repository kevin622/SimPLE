class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.change_idx = 0
    
    def __len__(self):
        return len(self.memory)

    def push(self, value):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.change_idx] = value
        self.change_idx = (self.change_idx + 1) % self.capacity
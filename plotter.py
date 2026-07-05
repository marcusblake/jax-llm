import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        self.data = []

    def add_step_results(self, results):
        self.data.append(results)
    
    def plot(self):
        pass
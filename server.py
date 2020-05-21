from model import *
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.ModularVisualization import ModularServer

class SigmoidText(TextElement):
    '''
    Display the Sigmoid Formula.
    '''
    def __init__(self):
        pass

    def render(self, model):
        return "Sigmoid Function: <math>σ(x) = 1&divide;(1 + e <sup>4 tanβ(x - α)</sup>)</math>"


def agent_portrayal(agent):
    portrayal = {"Shape": "arrowHead",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "red",
                 "scale": 0.5,
                 "heading_x": agent.heading[0],
                 "heading_y": agent.heading[1]}
    return portrayal

grid = CanvasGrid(agent_portrayal, 5, 5, 500, 500)
chart = ChartModule([{"Label": "Average Communicative Success",
                      "Color": "Black"}],
                    data_collector_name='datacollector')

# Number of Agents in the simulation
n = UserSettableParameter('slider', 'Number of agents', value=5, min_value=2, max_value=10, step=1)
# Word-meaning coupling change rate
r = UserSettableParameter('slider', 'Vocabulary change rate', value=5, min_value=1, max_value=10, step=1)
# Sigmoid variable
alpha = UserSettableParameter('slider', 'Alpha', value=0.49, min_value=0.0, max_value=1.0, step=0.01)
# Sigmoid variabl
beta = UserSettableParameter('slider', 'Beta', value=80, min_value=0.0, max_value=90, step=1)
sigmoid_text = SigmoidText()

server = ModularServer(LanguageModel,
                       [sigmoid_text, grid, chart],
                       "Language and Self-Organization",
                       {"n": n,
                        "r": r,
                        "alpha": alpha,
                        "beta": beta,
                        "width":5, "height":5})
server.port = 8521 # The default
# server.launch()

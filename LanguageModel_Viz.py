from LanguageModel import *
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "red",
                 "r": 0.5}
    return portrayal

grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)
# TODO: Add chart module to see metrics' evolution

server = ModularServer(LanguageModel,
                       [grid],
                       "Language and Self-Organization",
                       {"N":5, "width":10, "height":10})
server.port = 8521 # The default
server.launch()

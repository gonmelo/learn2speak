from LanguageModel import *
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule


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
server = ModularServer(LanguageModel,
                       [grid, chart],
                       "Language and Self-Organization",
                       {"N":5, "width":5, "height":5})
server.port = 8521 # The default
server.launch()

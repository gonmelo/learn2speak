from model import *
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement


class DialogText(TextElement):
    '''
    Display the Number of Dialogs.
    '''
    def __init__(self, var_name):
        self.var_name = var_name

    def render(self, model):
         return  "Occurred Dialogs: " + str(getattr(model, self.var_name))


class HistogramModule(VisualizationElement):
    package_includes = ["Chart.min.js"]
    local_includes = ["HistogramModule.js"]

    def __init__(self, bins, canvas_height, canvas_width):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.bins = bins
        new_element = "new HistogramModule({}, {}, {})"
        new_element = new_element.format(bins,
                                         canvas_width,
                                         canvas_height)
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        agents_success = [
            agent.comm_success for agent in model.schedule.agents]

        return agents_success


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
chart = ChartModule([{"Label": "Average Success (of the last window of dialogs)",
                      "Color": "Black"}],
                    data_collector_name='datacollector')

# Number of Agents in the simulation
n = UserSettableParameter('slider', 'Total Number of agents',
                          value=5, min_value=0, max_value=10, step=1)

l = UserSettableParameter('slider', 'Number of initial literate agents',
                          value=0, min_value=0, max_value=10, step=1)
# Word-meaning coupling change rate
r = UserSettableParameter(
    'slider', 'Vocabulary change rate', value=5, min_value=1, max_value=10, step=1)
# Sigmoid variable
# alpha = UserSettableParameter(
    # 'slider', 'Alpha', value=0.49, min_value=0.0, max_value=1.0, step=0.01)
# Sigmoid variabl
# beta = UserSettableParameter(
    # 'slider', 'Beta', value=80, min_value=0.0, max_value=90, step=1)
# Word creation probability
new_word_rate = UserSettableParameter(
    'slider', 'New word generation probability', value=0.05, min_value=0, max_value=1, step=0.01)
# Antecipated meaning probability
antecipated_prob = UserSettableParameter(
    'slider', 'Extralinguistic Initiation probability', value=0.5, min_value=0, max_value=1, step=0.01)
# Last success window size
success_window = UserSettableParameter(
    'slider', 'Communicative Success Window Size', value=50, min_value=5, max_value=100, step=1)

dialog_text = DialogText(var_name="total_dialogs")

alist = []
for i in range(10):
    alist.append("a-" + str(i+1))

histogram = HistogramModule(alist, 200, 500)

server = ModularServer(LanguageModel,
                       [dialog_text, grid, chart, histogram],
                       "Language and Self-Organization",
                       {"n": n,
                        "literate": l,
                        "r": r,
                        "alpha": 0.49,
                        "beta": 80,
                        "new_word_rate": new_word_rate,
                        "antecipated_prob": antecipated_prob,
                        "success_window": success_window,
                        "width": 5, "height": 5})
server.port = 8521  # The default
# server.launch()

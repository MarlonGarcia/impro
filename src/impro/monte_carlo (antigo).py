import panel as pn
pn.extension()



#%%

import numpy as np
import matplotlib.pyplot as plt
import panel as pn
import param

pn.extension()

class InteractiveFunction(param.Parameterized):
    n = param.Integer(default=5, bounds=(1, 20))
    cm = param.Number(default=50.0, bounds=(0, 100))
    
    def __init__(self, **params):
        super().__init__(**params)
        self.input_fields = []
        self.output_labels = []
        self.plot_pane = pn.pane.Matplotlib()
        self.param.watch(self.update_plot, ['n', 'cm'])
        self.create_inputs()
    
    def create_inputs(self):
        self.input_fields = [
            [pn.widgets.FloatInput(name=f'Val{i+1},{j+1}', value=0.0) for j in range(5)]
            for i in range(self.n)
        ]
        self.output_labels = [pn.pane.Markdown(f"Sum: 0") for _ in range(self.n)]
    
    def update_plot(self, *events):
        self.create_inputs()
        values = np.array([[widget.value for widget in row] for row in self.input_fields])
        sums = values.sum(axis=1)
        
        for i, row_sum in enumerate(sums):
            self.output_labels[i].object = f"Sum: {row_sum:.2f}"
        
        fig, ax = plt.subplots()
        ax.imshow(values, cmap='viridis', aspect='auto')
        plt.colorbar(ax.imshow(values, cmap='viridis', aspect='auto'))
        self.plot_pane.object = fig
    
    @param.depends('n', 'cm')
    def view(self):
        sliders = [pn.widgets.FloatSlider.from_param(self.param.cm)]
        inputs = [pn.Row(*row, self.output_labels[i]) for i, row in enumerate(self.input_fields)]
        buttons = pn.Row(
            pn.widgets.Button(name="Run the Simulation", button_type="primary"),
            pn.widgets.Button(name="More Information", button_type="default")
        )
        return pn.Column(
            pn.Param(self.param.n),
            pn.Param(self.param.cm),
            *inputs,
            buttons,
            self.plot_pane,
            *sliders
        )

interactive_function = InteractiveFunction()
interactive_function.view().servable()

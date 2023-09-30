import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

class ScatterPlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a central widget and set it as the main window's central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a layout for the central widget
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Create a PyQtGraph widget
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Generate some sample data
        num_points = 100
        self.x_data = np.random.rand(num_points)
        self.y_data = np.random.rand(num_points)

        # Create a scatter plot item
        self.scatter_plot_item = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))
        self.scatter_plot_item.setData(x=self.x_data, y=self.y_data)

        # Add the scatter plot item to the plot widget
        self.plot_widget.addItem(self.scatter_plot_item)

        # Set the plot axis labels
        self.plot_widget.setLabel('left', 'Y Axis')
        self.plot_widget.setLabel('bottom', 'X Axis')

        # Set the plot title
        self.plot_widget.setTitle('Scatter Plot Example')

        # Show grid lines
        self.plot_widget.showGrid(x=True, y=True)

        # Create a TextItem to show the value label when hovering over a point
        self.label = pg.TextItem(anchor=(0.5, 0), color=(0, 0, 255), fill=pg.mkBrush(255, 255, 255, 200))
        self.plot_widget.addItem(self.label)
        self.label.hide()

        # Connect the mouseMoved signal to a function that updates the label
        self.scatter_plot_item.scene().sigMouseMoved.connect(self.update_label)

    def update_label(self, event):
        # Get the mouse position in the plot coordinates
        pos = event  # Just use the event directly

        # Map the mouse position to data coordinates
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)

        # Find the nearest data point
        distance = np.sqrt((self.x_data - mouse_point.x()) ** 2 + (self.y_data - mouse_point.y()) ** 2)
        index = np.argmin(distance)

        # Update the label position and text
        self.label.setPos(self.x_data[index], self.y_data[index])
        self.label.setText(f'({self.x_data[index]:.2f}, {self.y_data[index]:.2f})')

        # Show the label
        self.label.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ScatterPlotWindow()
    window.setGeometry(100, 100, 800, 600)
    window.setWindowTitle('PyQtGraph Scatter Plot with Hover Label')
    window.show()
    sys.exit(app.exec_())

from tkinter import Tk, filedialog

import traitlets
from ipywidgets import widgets
import matplotlib.pyplot as plt

from comparer import compare, get_metric
from image_processing import process
from reconstructor import create_network
from utils import Load, draw_image, Img
from IPython.display import display, Markdown, Latex


class SelectFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self, label):
        super(SelectFilesButton, self).__init__()
        # Add the selected_files trait
        self.label = label
        self.add_traits(files=traitlets.traitlets.List())
        # Create the button.
        self.description = "Select " + label
        self.icon = "square-o"
        self.style.button_color = "orange"
        # Set on click behavior.
        self.on_click(self.select_files)
        self.image = None
        self.files = []

    @staticmethod
    def select_files(b):
        """Generate instance of tkinter.filedialog.
        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected fileswill be set to b.value
        try:
            b.files = filedialog.askopenfilename(multiple=True)
            b.image = Load.load_image(b.files[0])
            print("selected %s" % b.image.get_file_name())
            b.description = b.label + " Selected"
            b.icon = "check-square-o"
            b.style.button_color = "lightgreen"
        except Exception as e:
            print(e)


def show_images(x):
    plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    draw_image(image_select.image.original, 1, 3, 1, False)
    draw_image(manual_select.image.original, 1, 3, 2, False)
    draw_image(mask_select.image.original, 1, 3, 3)


image_select = SelectFilesButton("Image")
mask_select = SelectFilesButton("Mask")
manual_select = SelectFilesButton("Manual")

run_img_proc_btn = widgets.Button(
    description='Image processing',
    disabled=False,
    button_style='info',
    tooltip='Image processing',
    icon='check',
)

run_network_btn = widgets.Button(
    description='Conv. network',
    disabled=False,
    button_style='info',
    tooltip='Conv. network',
    icon='check',
)
show_images_btn = widgets.Button(
    description='Show Images',
    disabled=False,
    button_style='info',
    tooltip='Conv. network',
)


def run_image_proc(x):
    img, manual, mask, name = get_images()
    mark = process(img, mask)
    compare_results(manual, mark, mask, "image processing result")


def get_images():
    img = image_select.image.image
    manual = manual_select.image.image
    mask = mask_select.image.image
    name = image_select.image.get_file_name()
    return img, manual, mask, name


def run_network(x):
    network = create_network(70)
    img, manual, mask, name = get_images()
    mark = network.mark(img, mask, name)
    compare_results(manual, mark, mask, "network result")


def compare_results(manual, mark, mask, label):
    result, FN, FP, TN, TP, points = compare(manual, mask, mark)
    metric = get_metric(FN, FP, TN, TP, label, points)
    plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    draw_image(manual, 1, 3, 1, False)
    draw_image(mark, 1, 3, 2, False)
    draw_image(result, 1, 3, 3)
    display(Markdown(metric))


run_img_proc_btn.on_click(run_image_proc)
run_network_btn.on_click(run_network)
show_images_btn.on_click(show_images)
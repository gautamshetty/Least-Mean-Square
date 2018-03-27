import numpy as np
import Tkinter as Tk
import matplotlib
import scipy.misc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
from numpy import linalg as LA
from scipy import sparse

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import colorsys


class ClDataSet:
    # This class encapsulates the data set
    # The data set includes input samples and targets
    def __init__(self, samples=[[0., 0., 1., 1.], [0., 1., 0., 1.]], targets=[[0., 1., 1., 0.]]):
        # Note: input samples are assumed to be in column order.
        # This means that each column of the samples matrix is representing
        # a sample point
        # The default values for samples and targets represent an exclusive or
        self.samples = np.array(samples)

        if targets != None:
            self.targets = np.array(targets)
        else:
            self.targets = None


nn_experiment_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs": 2,  # number of inputs to the network
    "learning_rate": 0.1,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 3, "activation_function": "hardlimit"}],  # list of dictionaries
    "data_set": ClDataSet(),
    'number_of_classes': 3,
    'number_of_samples_in_each_class': 3,
    'sample_size': ClDataSet(),
    'x_coords':[],
    'y_coords':[],
    'delayed_terms': 1,
    'sample_size_percentage':100,
    'batch_size':20,
    'number_of_iterations':1
}


class ClNNExperiment:
    """
    This class presents an experimental setup for a single layer Perceptron
    """

    def __init__(self, settings={}):
        self.__dict__.update(nn_experiment_default_settings)
        self.__dict__.update(settings)
        # Set up the neural network
        settings = {"min_initial_weights": self.min_initial_weights,  # minimum initial weight
                    "max_initial_weights": self.max_initial_weights,  # maximum initial weight
                    "number_of_inputs": self.number_of_inputs,  # number of inputs to the network
                    "learning_rate": self.learning_rate,  # learning rate
                    "layers_specification": self.layers_specification,
                    "x_coords": self.x_coords,
                    "y_coords":self.y_coords,
                    "delayed_terms":self.delayed_terms,
                    "sample_size_percentage":self.sample_size_percentage,
                    "batch_size":self.batch_size,
                    "number_of_iterations":self.number_of_iterations
                    }
        self.neural_network = ClNeuralNetwork(self, settings)
        # Make sure that the number of neurons in the last layer is equal to number of classes
        self.neural_network.layers[-1].number_of_neurons = self.number_of_classes

    def run_forward_pass(self, display_input=True, display_output=True,
                         display_targets=True, display_target_vectors=True,
                         display_error=True):
        self.neural_network.calculate_output(self.data_set.samples)

        if display_input:
            print "Input : ", self.data_set.samples
        if display_output:
            print 'Output : ', self.neural_network.output
        if display_targets:
            print "Target (class ID) : ", self.target
        if display_target_vectors:
            print "Target Vectors : ", self.desired_target_vectors
        if self.desired_target_vectors.shape == self.neural_network.output.shape:
            self.error = self.desired_target_vectors - self.neural_network.output
            if display_error:
                print 'Error : ', self.error
        else:
            print "Size of the output is not the same as the size of the target.", \
                "Error cannot be calculated."

    def read_csv_as_matrix(self, file_name):
        # Each row of data in the file becomes a row in the matrix
        # So the resulting matrix has dimension [num_samples x sample_dimension]
        data = np.loadtxt(file_name, skiprows=1, delimiter=',', dtype=np.float32)
        return data

    """
        Read the excel sheet and populate the samples data set.
    """
    def create_samples(self):

        self.data_set.samples = self.read_csv_as_matrix("stock_data.csv")

        """
            Using the 1001 th element as target Price and Volume.
        """
        self.desired_target_vectors = self.data_set.samples[1001]

        self.data_set.sample_size = 1000 * self.sample_size_percentage / 100
        self.data_set.samples = self.data_set.samples[0:self.data_set.sample_size]

        print len(self.data_set.samples)
        print self.data_set.samples

        print "samples created"

    """
        Reset the target vectors and weights to initial values.
        Also reset the x & y co-ordinates to empty to reset the graph.
    """
    def reset(self):

        targetVectors = []

        for k in range(10):
            targetZeroVector = np.ones(self.neural_network.layers[-1].number_of_inputs_to_layer, int)
            targetZeroVector[k] = -1
            targetVectors.append(targetZeroVector)

        self.desired_target_vectors = targetVectors

        self.neural_network.layers[-1].number_of_neurons = self.number_of_classes = len(targetVectors)

        self.neural_network.randomize_weights()

        self.x_coords = []
        self.y_coords = []

    def adjust_weights(self):

        self.neural_network.adjust_weights(self.data_set.samples, self.data_set.sample_size,
                                           self.desired_target_vectors)

    def setWeightsToZero(self):
        self.neural_network.setWeightsToZero()

class ClNNGui2d:
    """
    This class presents an experiment to demonstrate
    Perceptron learning in 2d space.
    """

    def __init__(self, master, nn_experiment):
        self.master = master
        #
        self.nn_experiment = nn_experiment
        self.number_of_classes = self.nn_experiment.number_of_classes
        self.xmin = -2
        self.xmax = 2
        self.ymin = -2
        self.ymax = 2
        self.master.update()
        self.number_of_samples_in_each_class = self.nn_experiment.number_of_samples_in_each_class
        self.learning_rate = self.nn_experiment.learning_rate
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        self.step_size = 0.02
        self.current_sample_loss = 0
        self.sample_points = []
        self.target = []
        self.sample_colors = []
        self.weights = np.array([])
        self.class_ids = np.array([])
        self.output = np.array([])
        self.desired_target_vectors = np.array([])
        self.xx = np.array([])
        self.yy = np.array([])
        self.loss_type = ""
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")

        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)

        """self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = plt.subplot(111)"""

        """
            Show the graph with the initial x & y co-ordinates.
        """
        self.refreshGraph()

        """self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)"""

        # Create sliders frame
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='s1')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='s1')
        # Create buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')

        # Set up the sliders
        ivar = Tk.IntVar()
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.001, to_=1, resolution=0.01, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        ivar = Tk.IntVar()
        self.delayed_terms = 1
        self.delayed_terms_slider_label = Tk.Label(self.sliders_frame, text="Delayed Terms")
        self.delayed_terms_slider_label.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.delayed_terms_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0, to_=10, resolution=1, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.delayed_terms_slider_callback())
        self.delayed_terms_slider.set(self.delayed_terms)
        self.delayed_terms_slider.bind("<ButtonRelease-1>", lambda event: self.delayed_terms_slider_callback())
        self.delayed_terms_slider.grid(row=1, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        ivar = Tk.IntVar()
        self.sample_size_percentage = 100
        self.sample_size_percentage_slider_label = Tk.Label(self.sliders_frame, text="Sample Size Percentage")
        self.sample_size_percentage_slider_label.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.sample_size_percentage_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                                      from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                                      activebackground="#FF0000",
                                                      highlightcolor="#00FFFF", width=10,
                                                      command=lambda event: self.sample_size_percentage_slider_callback())
        self.sample_size_percentage_slider.set(self.sample_size_percentage)
        self.sample_size_percentage_slider.bind("<ButtonRelease-1>", lambda event: self.sample_size_percentage_slider_callback())
        self.sample_size_percentage_slider.grid(row=2, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        ivar = Tk.IntVar()
        self.batch_size = 20
        self.batch_size_slider_label = Tk.Label(self.sliders_frame, text="Batch Size")
        self.batch_size_slider_label.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.batch_size_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                          from_=0, to_=200, resolution=1, bg="#DDDDDD",
                                          activebackground="#FF0000",
                                          highlightcolor="#00FFFF", width=10,
                                          command=lambda
                                          event: self.batch_size_slider_callback())
        self.batch_size_slider.set(self.batch_size)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=3, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        ivar = Tk.IntVar()
        self.number_of_iterations = 1
        self.number_of_iterations_slider_label = Tk.Label(self.sliders_frame, text="Number of Iterations")
        self.number_of_iterations_slider_label.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.number_of_iterations_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                                    from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                                    activebackground="#FF0000",
                                                    highlightcolor="#00FFFF", width=10,
                                                    command=lambda
                                                    event: self.number_of_iterations_slider_callback())
        self.number_of_iterations_slider.set(self.number_of_iterations)
        self.number_of_iterations_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_iterations_slider_callback())
        self.number_of_iterations_slider.grid(row=4, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.print_nn_parameters_button = Tk.Button(self.buttons_frame,
                                                    text="Print NN Parameters",
                                                    bg="yellow", fg="red",
                                                    command=lambda: self.print_nn_parameters_button_callback())
        self.print_nn_parameters_button.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.set_weights_zero = Tk.Button(self.buttons_frame,
                                                  text="Set Weights to Zero",
                                                  bg="yellow", fg="red",
                                                  command=lambda: self.set_weights_zero_button_callback())
        self.set_weights_zero.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.initialize()

    def initialize(self):
        self.nn_experiment.create_samples()
        self.nn_experiment.neural_network.randomize_weights()

    def display_samples_on_image(self):
        # Display the samples for each class
        for class_index in range(0, self.number_of_classes):
            self.axes.scatter(self.nn_experiment.data_set.samples[0, class_index * self.number_of_samples_in_each_class: \
                (class_index + 1) * self.number_of_samples_in_each_class],
                              self.nn_experiment.data_set.samples[1, class_index * self.number_of_samples_in_each_class: \
                                  (class_index + 1) * self.number_of_samples_in_each_class],
                              c=self.sample_points_colors(class_index * (1.0 / self.number_of_classes)),
                              marker=(3 + class_index, 1, 0), s=50)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas.draw()

    def refresh_display(self):
        self.refreshGraph()

    """
        Displays a bar graph according to the x & y co-ordinates.
    """
    def refreshGraph(self):

        print "refresh display"

        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = plt.subplot(111)

        y_co_ord = self.nn_experiment.y_coords
        # std_men = (2, 3, 4, 1, 2)
        fig, ax = plt.subplots()

        x_co_ord = self.nn_experiment.x_coords
        bar_width = 0.10

        opacity = 0.4
        error_config = {'ecolor': '0.3'}

        error_graph = self.axes.bar(x_co_ord, y_co_ord, bar_width)

        print "bar graph plotted"
        print error_graph

        self.axes.set_xlabel('Iterations')
        self.axes.set_ylabel('Errors')
        self.axes.set_title('Graph of Errors in LMS Learning')
        self.axes.legend()

        self.axes.relim()
        plt.draw()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

    def display_output_nodes_net_boundaries(self):
        output_layer = self.nn_experiment.neural_network.layers[-1]
        for output_neuron_index in range(output_layer.number_of_neurons):
            w1 = output_layer.weights[output_neuron_index][0]
            w2 = output_layer.weights[output_neuron_index][1]
            w3 = output_layer.weights[output_neuron_index][2]
            if w1 == 0 and w2 == 0:
                data = [(0, 0), (0, 0), 'r']
            elif w1 == 0:
                data = [(self.xmin, self.xmax), (float(w3) / w2, float(w3) / w2), 'r']
            elif w2 == 0:
                data = [(float(-w3) / w1, float(-w3) / w1), (self.ymin, self.ymax), 'r']
            else:
                data = [(self.xmin, self.xmax),  # in form of (x1, x2), (y1, y2)
                        ((-w3 - float(w1 * self.xmin)) / w2,
                         (-w3 - float(w1 * self.xmax)) / w2), 'r']
            self.axes.plot(*data)

    def learning_rate_slider_callback(self):
        self.learning_rate = self.learning_rate_slider.get()
        self.nn_experiment.learning_rate = self.learning_rate
        self.nn_experiment.neural_network.learning_rate = self.learning_rate
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        self.refresh_display()

    def delayed_terms_slider_callback(self):
        self.nn_experiment.delayed_terms = self.delayed_terms

    def sample_size_percentage_slider_callback(self):
        self.nn_experiment.sample_size_percentage = self.sample_size_percentage

    def batch_size_slider_callback(self):
        self.nn_experiment.batch_size = self.batch_size

    def number_of_iterations_slider_callback(self):
        self.nn_experiment.number_of_iterations = self.number_of_iterations

    def number_of_classes_slider_callback(self):
        self.number_of_classes = self.number_of_classes_slider.get()
        self.nn_experiment.number_of_classes = self.number_of_classes
        self.nn_experiment.neural_network.layers[-1].number_of_neurons = self.number_of_classes
        self.nn_experiment.neural_network.randomize_weights()
        self.initialize()
        self.refresh_display()

    def number_of_samples_slider_callback(self):
        self.number_of_samples_in_each_class = self.number_of_samples_slider.get()
        self.nn_experiment.number_of_samples_in_each_class = self.number_of_samples_slider.get()
        self.nn_experiment.create_samples()
        self.refresh_display()

    def create_new_samples_bottun_callback(self):
        temp_text = self.create_new_samples_bottun.config('text')[-1]
        self.create_new_samples_bottun.config(text='Please Wait')
        self.create_new_samples_bottun.update_idletasks()
        self.nn_experiment.create_samples()
        self.refresh_display()
        self.create_new_samples_bottun.config(text=temp_text)
        self.create_new_samples_bottun.update_idletasks()

    def adjust_weights_button_callback(self):
        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')
        for k in range(self.number_of_iterations):
            self.nn_experiment.adjust_weights()
            self.refresh_display()
        self.adjust_weights_button.config(text=temp_text)
        self.adjust_weights_button.update_idletasks()

    """def learning_method_dropdown_callback(self):
        self.learning_method = self.learning_method_variable.get()"""

    def randomize_weights_button_callback(self):
        temp_text = self.randomize_weights_button.config('text')[-1]
        self.randomize_weights_button.config(text='Please Wait')
        self.randomize_weights_button.update_idletasks()
        self.nn_experiment.neural_network.randomize_weights()
        # self.nn_experiment.neural_network.display_network_parameters()
        # self.nn_experiment.run_forward_pass()
        self.refresh_display()
        self.randomize_weights_button.config(text=temp_text)
        self.randomize_weights_button.update_idletasks()

    def print_nn_parameters_button_callback(self):
        temp_text = self.print_nn_parameters_button.config('text')[-1]
        self.print_nn_parameters_button.config(text='Please Wait')
        self.print_nn_parameters_button.update_idletasks()
        self.nn_experiment.neural_network.display_network_parameters()
        self.refresh_display()
        self.print_nn_parameters_button.config(text=temp_text)
        self.print_nn_parameters_button.update_idletasks()

    def reset_button_callback(self):

        self.nn_experiment.reset()
        self.refresh_display()

    def set_weights_zero_button_callback(self):
        self.nn_experiment.setWeightsToZero()

neural_network_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs": 2,  # number of inputs to the network
    "learning_rate": 0.001,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 3,
                              "activation_function": "hardlimit"}]  # list of dictionaries
}


class ClNeuralNetwork:
    """
    This class presents a multi layer neural network
    """

    def __init__(self, experiment, settings={}):
        self.__dict__.update(neural_network_default_settings)
        self.__dict__.update(settings)
        # create nn
        self.experiment = experiment
        self.layers = []
        self.input_index = 0
        for layer_index, layer in enumerate(self.layers_specification):
            if layer_index == 0:
                layer['number_of_inputs_to_layer'] = self.number_of_inputs
            else:
                layer['number_of_inputs_to_layer'] = self.layers[layer_index - 1].number_of_neurons
            self.layers.append(ClSingleLayer(layer))

    def randomize_weights(self, min=-0.1, max=0.1):
        # randomize weights for all the connections in the network
        for layer in self.layers:
            layer.randomize_weights(self.min_initial_weights, self.max_initial_weights)

    def display_network_parameters(self, display_layers=True, display_weights=True):
        for layer_index, layer in enumerate(self.layers):
            print "\n--------------------------------------------", \
                "\nLayer #: ", layer_index, \
                "\nNumber of Nodes : ", layer.number_of_neurons, \
                "\nNumber of inputs : ", self.layers[layer_index].number_of_inputs_to_layer, \
                "\nActivation Function : ", layer.activation_function, \
                "\nWeights : ", layer.weights

    def calculate_output(self, input_values):
        # Calculate the output of the network, given the input signals
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                output = layer.calculate_output(input_values)
            else:
                output = layer.calculate_output(output)
        self.output = output
        return self.output

    """
        Reset weights to zero.
    """
    def setWeightsToZero(self):

        for layer_index, layer in enumerate(self.layers):
            layer.weights = [0, 0]

    """
        Adjust weights according to LMS learning rule.
    """
    def adjust_weights(self, input_samples, sample_size, desired_target_vectors):

        for layer_index, layer in enumerate(self.layers):

            for sample_index in range(self.experiment.batch_size):

                """
                    Calculate actual output
                """
                actual_output =  np.multiply(layer.weights, input_samples[self.input_index])
                error = np.subtract(desired_target_vectors, actual_output)

                """
                    Calculate weights using LMS learning.
                """
                weight_input = np.multiply(2, error)
                weight_input = np.multiply(self.experiment.learning_rate, weight_input)

                weight_input = np.multiply(input_samples[self.input_index], weight_input)
                layer.weights = np.add(layer.weights, weight_input)

                self.input_index = (self.input_index + 1) % sample_size

            mean_sq_err_array = []
            max_error = [0, 0]
            for index in range(self.batch_size):

                """
                    Delay the terms as per the slider delayed terms.
                """
                if index == 0 & index < self.experiment.delayed_terms:
                    input = input_samples[index]
                if index != 0 & index < self.experiment.delayed_terms:
                    input = input_samples[index - 1]
                else:
                    input = input_samples[index]

                """
                    Calculate actual output and mean square error.
                """
                actual_output = np.multiply(layer.weights, input)
                error_sq = np.subtract(desired_target_vectors, actual_output)
                error_sq = np.square(error_sq)
                mean_sq_err_array.append(error_sq)

                if (max_error[0] < error_sq[0]):
                    max_error[0] = error_sq[0]

                if (max_error[1] < error_sq[1]):
                    max_error[1] = error_sq[1]

            mean_sq_error = np.mean(mean_sq_err_array)

            """
                Set the x & y co-ordinates to be shown on the bar graph.
            """
            self.experiment.x_coords.append(self.input_index)
            self.experiment.y_coords.append(mean_sq_error)

            print mean_sq_error

        print "X Co-ords", self.experiment.x_coords
        print "Y Co-ords", self.experiment.y_coords

single_layer_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs_to_layer": 2,  # number of input signals
    "number_of_neurons": 2,  # number of neurons in the layer
    "activation_function": "hardlimit"  # default activation function
}


class ClSingleLayer:
    """
    This class presents a single layer of neurons
    """

    def __init__(self, settings):
        self.__dict__.update(single_layer_default_settings)
        self.__dict__.update(settings)
        self.randomize_weights()

    def randomize_weights(self, min_initial_weights=None, max_initial_weights=None):
        if min_initial_weights == None:
            min_initial_weights = self.min_initial_weights
        if max_initial_weights == None:
            max_initial_weights = self.max_initial_weights
        self.weights = np.zeros(self.number_of_neurons, int)#np.random.uniform(min_initial_weights, max_initial_weights, (self.number_of_neurons, self.number_of_inputs_to_layer + 1))

    def calculate_output(self, input_values):
        # Calculate the output of the layer, given the input signals
        # NOTE: Input is assumed to be a column vector. If the input
        # is given as a matrix, then each column of the input matrix is assumed to be a sample
        if len(input_values.shape) == 1:
            net = self.weights.dot(np.append(input_values, 1))
        else:
            net = self.weights.dot(np.vstack([input_values, np.ones((1, input_values.shape[1]), float)]))
        if self.activation_function == 'linear':
            self.output = net
        if self.activation_function == 'sigmoid':
            self.output = sigmoid(net)
        if self.activation_function == 'hardlimit':
            np.putmask(net, net > 0, 1)
            np.putmask(net, net <= 0, 0)
            self.output = net
        return self.output


if __name__ == "__main__":
    nn_experiment_settings = {
        "min_initial_weights": -0.1,  # minimum initial weight
        "max_initial_weights": 0.1,  # maximum initial weight
        "number_of_inputs": 2,  # number of inputs to the network
        "learning_rate": 0.1,  # learning rate
        "layers_specification": [{"number_of_neurons": 3, "activation_function": "hardlimit"}],  # list of dictionaries
        "data_set": ClDataSet(),
        'number_of_classes': 2,
        'number_of_samples_in_each_class': 3
    }
    np.random.seed(1)
    ob_nn_experiment = ClNNExperiment(nn_experiment_settings)
    main_frame = Tk.Tk()
    main_frame.title("Hebbian Learning")
    main_frame.geometry('640x480')
    ob_nn_gui_2d = ClNNGui2d(main_frame, ob_nn_experiment)
    main_frame.mainloop()

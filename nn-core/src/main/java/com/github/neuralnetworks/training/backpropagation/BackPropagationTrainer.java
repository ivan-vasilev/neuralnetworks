package com.github.neuralnetworks.training.backpropagation;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.LayerCalculator;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.UniqueList;

public class BackPropagationTrainer extends OneStepTrainer<NeuralNetwork> {

    private Map<Layer, Matrix> activations;
    private Map<Layer, Matrix> backpropagation;

    public BackPropagationTrainer() {
	super();
	activations = new HashMap<Layer, Matrix>();
    }

    public BackPropagationTrainer(Properties properties) {
	super(properties);
	activations = new HashMap<Layer, Matrix>();
	backpropagation = new HashMap<Layer, Matrix>();
    }

    @Override
    protected void learnInput(TrainingInputData data) {
	NeuralNetwork nn = getNeuralNetwork();
	Set<Layer> calculatedLayers = new UniqueList<Layer>();

	activations.put(nn.getInputLayer(), data.getInput());
	calculatedLayers.add(nn.getInputLayer());
	LayerCalculator lc = getLayerCalculator();
	lc.calculate(calculatedLayers, activations, nn.getOutputLayer());

	OutputErrorDerivative d = getProperties().getParameter(Constants.OUTPUT_ERROR_DERIVATIVE);
	Matrix outputErrorDerivative = d.getOutputErrorDerivative(activations.get(nn.getOutputLayer()), data.getTarget());
	backpropagation.put(nn.getOutputLayer(), outputErrorDerivative);
	calculatedLayers.clear();
	calculatedLayers.add(nn.getOutputLayer());
	BackPropagationLayerCalculator blc = getProperties().getParameter(Constants.BACKPROPAGATION);
	blc.backpropagate(calculatedLayers, activations, backpropagation, nn.getInputLayer());
    }
}

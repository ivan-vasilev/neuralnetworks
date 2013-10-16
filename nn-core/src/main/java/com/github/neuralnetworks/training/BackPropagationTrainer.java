package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

public class BackPropagationTrainer extends Trainer<NeuralNetwork> {

    private Map<Layer, Matrix> results;

    public BackPropagationTrainer() {
	super();
	init();
    }

    public BackPropagationTrainer(Properties properties) {
	super(properties);
	init();
    }

    protected void init() {
	results = new HashMap<Layer, Matrix>();
    }

    @Override
    protected void learnInput(TrainingInputData data) {
	NeuralNetwork nn = getNeuralNetwork();

	results.put(nn.getInputLayer(), data.getInput());
	Set<Layer> calculatedLayers = new HashSet<Layer>();
	calculatedLayers.add(nn.getInputLayer());
	getLayerCalculator().calculate(calculatedLayers, results, nn.getOutputLayer());

	BackPropagation bp = getProperties().getParameter(Constants.BACKPROPAGATION);
	Matrix outputErrorDerivative = bp.getOutputErrorDerivative(results.get(nn.getOutputLayer()), data.getTarget());
	bp.backPropagate(results, outputErrorDerivative, nn);
    }
}

package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.LayerCalculator;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

public class BackPropagationTrainer extends Trainer<NeuralNetwork> {

    private Map<Layer, Matrix> results;

    public BackPropagationTrainer() {
	super();
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

	results.put(nn.getInputLayer(), data.getConvertedInput());
	Set<Layer> calculatedLayers = new HashSet<Layer>();
	calculatedLayers.add(nn.getInputLayer());
	LayerCalculator forwardCalculator = getProperties().getParameter(Constants.FORWARD_CALCULATOR);
	forwardCalculator.calculate(calculatedLayers, results, nn.getOutputLayer());

	BackPropagation bp = getProperties().getParameter(Constants.BACKPROPAGATION);
	bp.backPropagate(bp.getOutputErrorDerivative(results.get(nn.getOutputLayer()), data.getConvertedTarget()), nn);
    }
}

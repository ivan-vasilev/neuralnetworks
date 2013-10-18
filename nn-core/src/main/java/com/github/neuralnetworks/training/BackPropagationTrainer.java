package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.LayerCalculator;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.UniqueList;

public class BackPropagationTrainer extends Trainer<NeuralNetwork> {

    private Map<Layer, Matrix> results;

    public BackPropagationTrainer() {
	super();
	results = new HashMap<Layer, Matrix>();
    }

    public BackPropagationTrainer(Properties properties) {
	super(properties);
	results = new HashMap<Layer, Matrix>();
    }

    @Override
    protected void learnInput(TrainingInputData data) {
	NeuralNetwork nn = getNeuralNetwork();

	results.put(nn.getInputLayer(), data.getInput());
	Set<Layer> calculatedLayers = new UniqueList<Layer>();
	calculatedLayers.add(nn.getInputLayer());
	LayerCalculator lc = getLayerCalculator();
	lc.calculate(calculatedLayers, results, nn.getOutputLayer());

	BackPropagation bp = getProperties().getParameter(Constants.BACKPROPAGATION);
	Matrix outputErrorDerivative = bp.getOutputErrorDerivative(results.get(nn.getOutputLayer()), data.getTarget());
	bp.backPropagate(results, outputErrorDerivative, nn);
    }
}
